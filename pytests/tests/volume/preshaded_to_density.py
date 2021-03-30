import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import tqdm
import time
import h5py
import argparse
import json
import subprocess
from collections import defaultdict

sys.path.append(os.getcwd())
from diffdvr import renderer_dtype_torch, renderer_dtype_np, \
    VolumeDensities, SmoothnessPrior
import pyrenderer

def optim():
    # Settings
    parser = argparse.ArgumentParser(
        description="Reconstruct a density volume from a color volume",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help="Input .hdf5 file")
    parser.add_argument('output', type=str, help="Output .hdf5 file")
    parser.add_argument('-e', '--epoch', type=int, default=-1,
                        help="The epoch of the color volume to use. Default -1, the last iteration")
    parser.add_argument('-s', '--scaling', type=float, default=2,
                        help="Scaling factor to the input volume before reconstruction")
    parser.add_argument('-a', '--alpha', type=float, default=1, help="""
        Extra weighting of the data term with the opacity between 0 and 1.
        0: no weighting
        1: full weighting, zero opacity leads to zero data term
        """)
    parser.add_argument('-b', '--beta', type=float, default=0.01, help="""
        Spatial smoothing prior
        """)
    parser.add_argument('--smoothOnly', action='store_true',
                        help="Perform only smoothing, best used together with '-I best-fit'")
    parser.add_argument('-I', '--initial', default="gauss",
                              choices=["gauss", "sphere", "best-fit"], help="""
                How the volume should be initialized:
                 - gauss: uniform, independent gaussian noise
                 - sphere: sphere density, one at the center, zero at the border
                 - best-fit: sample best fitting density                               
                """)
    parser.add_argument('--fitN', default=256, type=int,
                        help="The number of samples for best-fit sampling")
    parser.add_argument('-ow', '--fitOpacityWeighting', default=-1.0, type=float, help="""
        The opacity weighting for best-fit sampling. 
        Special value -1: weight = 1 / max opacity 
        """)
    parser.add_argument('-fi', '--fitIterations', default=1, type=int, help="""
        Number of times the fitting should take place.
        If greater one, differences to neighbor densities are taken into account.
        The strength is controlled by 'fitNeighborWeighting'.
        """)
    parser.add_argument('-nw', '--fitNeighborWeighting', default=0.1, type=float,
                        help="Weighting of the neighbors for multi-iteration fitting")
    parser.add_argument('-o', '--optimizer', default='Adam', type=str,
                              help="The optimizer class, 'torch.optim.XXX'")
    parser.add_argument('-lr', default=0.8, type=float, help="The learning rate")
    parser.add_argument('-i', '--iterations', default=50, type=int,
                              help="The number of iterations in the training")
    parser.add_argument('--optimParams', default="{}", type=str,
                              help="Additional optimizer parameters parsed as json")
    parser.add_argument('--seed', type=int, default=124, help='random seed to use. Default=124')
    parser.add_argument('--noCuda', action='store_true', help='Disable cuda')
    parser.add_argument('--volumeSaveFrequency', type=int, default=10,
                              help="The frequency at which to save the volume")

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.set_num_threads(4)

    device = torch.device("cpu") if opt.noCuda else torch.device("cuda")

    print("Working directory:", os.getcwd())
    print("Load input", os.path.join(os.getcwd(), opt.input))
    with h5py.File(os.path.join(os.getcwd(), opt.input), 'r') as f:
        assert 'volumes' in f, "no volume found in input file"
        color_volume_dset = f['volumes']
        assert len(color_volume_dset.shape) == 5, "color volume must be 5D"
        assert color_volume_dset.shape[1] == 4, "volume is not a color volume, invalid chanel count"
        assert -color_volume_dset.shape[0] <= opt.epoch < color_volume_dset.shape[0], "epoch is out of bounds"
        color_volume = f['volumes'][opt.epoch, ...]

        tf = f['reference_tf'][...]
        tf_mode_str = f.attrs['tfmode']
        if tf_mode_str == "texture":
            tf_mode = pyrenderer.TFMode.Texture
        if tf_mode_str == "gauss":
            tf_mode = pyrenderer.TFMode.Gaussian
        else:
            # other TF modes should probably work, but are not tested
            raise ValueError("so far, only texture TFs are supported, not " + tf_mode_str)
        settings_file_str = f.attrs['settingsFile']

    # to torch
    print("Input shape:", color_volume.shape)
    color_volume = torch.from_numpy(color_volume).to(device=device, dtype=renderer_dtype_torch)
    if opt.scaling != 1:
        color_volume = torch.nn.functional.interpolate(
            color_volume.unsqueeze(0), scale_factor=opt.scaling)[0]
    tf = torch.from_numpy(tf).to(device=device, dtype=renderer_dtype_torch)

    C, X, Y, Z = color_volume.shape

    print("Define loss")
    class Loss(torch.nn.Module):
        def __init__(self, color_volume, tf, tf_mode, alpha, beta):
            super().__init__()
            assert 0 <= alpha <= 1
            assert 0 <= beta
            self._alpha = alpha
            self._beta = beta
            self._color_volume = color_volume
            self._tf = tf
            self._tf_mode = tf_mode
            self._prior = SmoothnessPrior((1, 2, 3), reduction="sum" if opt.smoothOnly else "mean")
            self._weight = (1 - self._alpha) + self._alpha * torch.clamp(color_volume[3:4, :, :, :], 0, 1)

        def forward(self, density_volume):
            # data loss
            color = pyrenderer.TFUtils.preshade_volume(
                density_volume, self._tf, self._tf_mode)
            # diff = torch.linalg.norm(color-self._color_volume, dim=0, keepdim=True)
            diff = F.mse_loss(color, self._color_volume, reduction='none')
            data_loss = torch.mean(diff * self._weight)
            # smoothness prior
            prior_loss = self._prior(density_volume)
            # total loss
            if opt.smoothOnly:
                total_loss = prior_loss
            else:
                total_loss = data_loss + self._beta * prior_loss
            return total_loss, {
                'data_loss': data_loss.item(),
                'prior_loss': prior_loss.item()
            }

        def loss_names(self):
            return ["data_loss", "prior_loss"]

    loss = Loss(color_volume, tf, tf_mode, opt.alpha, opt.beta)
    loss.to(device=device)

    # initialize optimization
    epochs_fitting = opt.fitIterations
    epochs_fitting_with_volume = int(np.ceil(epochs_fitting / opt.volumeSaveFrequency))
    epochs_fitting = opt.volumeSaveFrequency * epochs_fitting_with_volume
    epochs_smoothing = opt.iterations
    epochs_smoothing_with_volume = int(np.ceil(epochs_smoothing / opt.volumeSaveFrequency))
    epochs_smoothing = opt.volumeSaveFrequency * epochs_smoothing_with_volume

    output_file = os.path.join(os.getcwd(), opt.output)
    print("Create output file", output_file)
    os.makedirs(os.path.split(output_file)[0], exist_ok=True)
    torch.set_grad_enabled(False)
    with h5py.File(output_file, 'w') as hdf5_file:
        for k, v in vars(opt).items():
            hdf5_file.attrs[k] = v
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            hdf5_file.attrs['git'] = git_commit
            print("git commit", git_commit)
        except:
            print("unable to get git commit")
        hdf5_file.attrs['settingsFile'] = settings_file_str
        hdf5_file.attrs['tfmode'] = tf_mode_str

        hdf5_file.create_dataset(
            "reference_tf", data=tf.cpu().numpy())
        times = hdf5_file.create_dataset("times", (max(2, epochs_fitting+epochs_smoothing),), dtype=renderer_dtype_np)
        volumes = hdf5_file.create_dataset(
            "volumes", (epochs_smoothing_with_volume + epochs_fitting_with_volume + 2, 1, X, Y, Z),
            dtype=renderer_dtype_np, chunks=(1, 1, X, Y, Z),
            compression="gzip")
        losses = dict([
            (name, hdf5_file.create_dataset(name, (max(2, epochs_fitting+epochs_smoothing),), dtype=renderer_dtype_np))
            for name in loss.loss_names() + ["total"]
        ])
        volume_iteration = 0
        loss_time_iteration = 0

        print("Create initial density volume")
        volume_densities = VolumeDensities()
        if opt.initial == "gauss":
            initial_volume_parameterized = torch.randn(
                1, X, Y, Z, dtype=renderer_dtype_torch, device=device)
        elif opt.initial == "sphere":
            _initial_volume_class = pyrenderer.Volume.create_implicit(
                pyrenderer.ImplicitEquation.Sphere, X)
            if opt.noCuda:
                _initial_volume_data = _initial_volume_class.getDataCpu(0)
            else:
                _initial_volume_class.copy_to_gpu()
                _initial_volume_data = _initial_volume_class.getDataGpu(0)
            _volume_min = torch.min(_initial_volume_data).item()
            _volume_max = torch.max(_initial_volume_data).item()
            _new_min = 1e-3
            _new_max = 1 - 1e-3
            # transform to _new_min/max
            _initial_volume_data = _new_min + (_new_max - _new_min) * (_initial_volume_data - _volume_min) / (
                        _volume_max - _volume_min)
            initial_volume_parameterized = VolumeDensities.prepare_input(
                _initial_volume_data)
            initial_volume_parameterized = F.interpolate(
                initial_volume_parameterized.unsqueeze(0), (X, Y, Z),
                mode='trilinear')[0]
        elif opt.initial == "best-fit":
            _opacity_weight = opt.fitOpacityWeighting
            if _opacity_weight < 0:
                _max_opacity = torch.max(color_volume[3, ...]).item()
                print("max opacity:", _max_opacity)
                _opacity_weight = (1.0 / _max_opacity) * (-_opacity_weight)
            _initial_volume_data = pyrenderer.TFUtils.find_best_fit(
                color_volume, tf, tf_mode, opt.fitN, _opacity_weight)
            print(tf.cpu().numpy())
            pyrenderer.sync()
            with torch.no_grad():
                if opt.fitIterations > 1:
                    _last_volume = _initial_volume_data
                    print("Iterative fitting")
                    start_time = time.time()
                    with tqdm.tqdm(epochs_fitting) as iteration_bar:
                        for iteration in range(epochs_fitting):
                            partial_losses = defaultdict(float)
                            total_loss, lx = loss(_last_volume)
                            for k, v in lx.items():
                                partial_losses[k] += v
                            partial_losses['total'] = total_loss.item()
                            end_time = time.time()
                            times[loss_time_iteration] = end_time - start_time
                            if iteration % opt.volumeSaveFrequency == 0:
                                volumes[volume_iteration] = _last_volume.detach().cpu().numpy()
                                volume_iteration += 1
                            for k, v in partial_losses.items():
                                losses[k][loss_time_iteration] = v
                            loss_time_iteration += 1

                            iteration_bar.update(1)
                            iteration_bar.set_description("Loss: %7.5f" % partial_losses['total'])

                            _initial_volume_data = pyrenderer.TFUtils.find_best_fit(
                                color_volume, tf, tf_mode, opt.fitN, _opacity_weight,
                                _last_volume, opt.fitNeighborWeighting)
                            pyrenderer.sync()
                            _last_volume = _initial_volume_data

                initial_volume_parameterized = VolumeDensities.prepare_input(
                    torch.clamp(_initial_volume_data, 1e-4, 1 - 1e-4))
        else:
            raise ValueError("Unknown initial volume setting: " + opt.initial)

        if opt.smoothOnly:
            print("Reset all voxels with opacity<10% of max opacity to zero")
            _max_opacity = torch.max(color_volume[3, ...]).item()
            smooth_only_mask_bool = color_volume[3:4, ...] > 0.1 * _max_opacity
            initial_volume_parameterized = torch.where(
                smooth_only_mask_bool, initial_volume_parameterized,
                torch.tensor([[[[-10.0]]]], dtype=renderer_dtype_torch, device=device))

        # initial volume
        last_volume = volume_densities(initial_volume_parameterized)
        volumes[volume_iteration] = last_volume.detach().cpu().numpy()
        volume_iteration += 1

        # optimization
        optimizer_class = getattr(torch.optim, opt.optimizer)
        optimizer_parameters = json.loads(opt.optimParams)
        optimizer_parameters['lr'] = opt.lr
    
        current_volume_parameterized = initial_volume_parameterized.clone()
        current_volume_parameterized.requires_grad_(True)
        optimizer = optimizer_class([current_volume_parameterized], **optimizer_parameters)

        # optimize
        last_loss = None
        last_volume = None

        def apply_boundary_conditions(current: torch.Tensor):
            if opt.smoothOnly:
                return torch.where(
                    smooth_only_mask_bool, initial_volume_parameterized,
                    current)
            else:
                return current

        def optim_closure():
            nonlocal last_loss, last_volume
            optimizer.zero_grad()
            partial_losses = defaultdict(float)
            volume = volume_densities(apply_boundary_conditions(current_volume_parameterized))
            volume.retain_grad()
            total_loss, lx = loss(volume)
            for k, v in lx.items():
                partial_losses[k] += v
            partial_losses['total'] = total_loss.item()

            # test
            total_loss.retain_grad()

            last_volume = volume.detach()
            last_loss = partial_losses
            pyrenderer.sync()
            total_loss.backward()
            pyrenderer.sync()
            return total_loss

        start_time = time.time()
        with tqdm.tqdm(epochs_smoothing) as iteration_bar:
            for iteration in range(epochs_smoothing):
                optimizer.step(optim_closure)

                end_time = time.time()
                times[loss_time_iteration] = end_time - start_time
                if iteration % opt.volumeSaveFrequency == 0:
                    volumes[volume_iteration] = last_volume.detach().cpu().numpy()
                    volume_iteration += 1
                for k, v in last_loss.items():
                    losses[k][loss_time_iteration] = v
                loss_time_iteration += 1

                iteration_bar.update(1)
                iteration_bar.set_description("Loss: %7.5f" % last_loss['total'])
        last_volume = volume_densities(apply_boundary_conditions(current_volume_parameterized))
        volumes[volume_iteration] = last_volume.detach().cpu().numpy()

    print("Done")

if __name__ == '__main__':
    optim()
