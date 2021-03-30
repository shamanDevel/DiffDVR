"""
Large hyperparameter training session
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
import os
import tqdm
import time
import h5py
import argparse
import json
from collections import defaultdict
import subprocess
import contextlib

@contextlib.contextmanager
def dummy_context_mgr():
    yield None

from diffdvr import Renderer, CameraOnASphere, Settings, setup_default_settings, \
    fibonacci_sphere, renderer_dtype_torch, renderer_dtype_np, VolumeDensities, \
    TfTexture, SmoothnessPrior, toCHW, VolumePreshaded
from losses import LossBuilder
import pyrenderer

class Config:
    def __init__(self):
        pass

    def init_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('outputFile', type=str, help="Output .hdf5 file")
        # dataset
        parser_group = parser.add_argument_group("Data")
        parser.add_argument('settingsFile', type=str, help="Settings .json file")
        parser_group.add_argument('-v', '--views', default=8, type=int, help="Number of views")
        parser_group.add_argument('-r', '--imgres', default=512, type=int, help="Image resolution")
        parser_group.add_argument('--minOpacity', default=0, type=float,
                                  help="Forces a minimal opacity on the transfer function")
        parser_group.add_argument('-tf', '--tfmode', default='texture',
                                  choices=['identity', 'texture', 'linear', 'gauss'])
        parser_group.add_argument('--opacityScaling', default=3, type=float,
                                  help="Opacity scaling, only used for identity TF")
        parser_group.add_argument('--randomizeTF', action='store_true',
                                  help="Use random TFs instead of the single TF from the settings")
        # model
        parser_group = parser.add_argument_group("Model")
        parser_group.add_argument('-R', '--volres', default=128, type=int, help="The volume resolution")
        parser_group.add_argument('--multiscale', default=-1, type=int, help="""
            If specified with an argument > 0, enables multi-scale optimization.
            The optimization starts with a low resolution volume of size specified by 'multiscale'
            and then doubles the resolution after 'iterations' epochs until the target
            resolution specified by 'volres' is reached.
            
            Note, there must be an integer n such that volres=multiscale**n .  
            """)
        parser_group.add_argument('-I', '--initial', default="gauss",
                                  choices=["gauss", "sphere", "warp", "file"], help="""
            How the volume should be initialized:
             - gauss: uniform, independent gaussian noise
             - sphere: sphere density, one at the center, zero at the border
             - warp: warp the original volume using perlin noise. See warpOptions
             - file: initialize with density volume from an external file                               
            """)
        parser_group.add_argument('--warpOptions', type=str,
                                  default="scale=0.01;res=4;octaves=4",
                                  help="Parameters of the perlin warping")
        parser_group.add_argument('--initialFilePath', type=str, default=None,
                                  help="Path to the .hdf5 file for the initial volume")
        parser_group.add_argument('--initialFileEpoch', type=int, default=-1,
                                  help="The epoch to take as the initial volume")
        parser_group.add_argument('--initialGaussMean', type=float, default=0,
                                  help="Mean of the gaussian for 'gauss' initialization. Beware of sigmoid parametrization!")
        parser_group.add_argument('--initialGaussStd', type=float, default=1,
                                  help="Standard deviation of the gaussian for 'gauss' initialization. Beware of sigmoid parametrization!")
        parser_group.add_argument('--preshaded', action='store_true',
                                  help="Uses a preshaded volume")
        # losses
        parser_group = parser.add_argument_group("Loss")
        parser_group.add_argument('-l1', default=0, type=float,
                                  help="Weight of the L1 image loss")
        parser_group.add_argument('-l2', default=0, type=float,
                                  help="Weight of the L2 image loss")
        parser_group.add_argument('-dssim', default=0, type=float,
                                  help="Weight of the DSSIM image loss")
        parser_group.add_argument('-lpips', default=0, type=float,
                                  help="Weight of the LPIPS image loss")
        parser_group.add_argument('-ps', '--priorSmoothing', default=0, type=float,
                                  help="Weight of the smoothing prior")
        # training
        parser_group = parser.add_argument_group("Training")
        parser_group.add_argument('-o', '--optimizer', default='Adam', type=str,
                                  help="The optimizer class, 'torch.optim.XXX'")
        parser_group.add_argument('-lr', default=0.8, type=float, help="The learning rate")
        parser_group.add_argument('-i', '--iterations', default=50, type=int,
                                  help="The number of iterations in the training")
        parser_group.add_argument('--optimParams', default="{}", type=str,
                                  help="Additional optimizer parameters parsed as json")
        parser_group.add_argument('-b', '--batches', default=4, type=int,
                                  help="Batch size for training")
        parser_group.add_argument('-bm', '--batchMode', default='full', choices=['full', 'stochastic'],
                                  help="Batch mode (full or stochastic gradient descent)")
        parser_group.add_argument('--seed', type=int, default=124, help='random seed to use. Default=124')
        parser_group.add_argument('--noCuda', action='store_true', help='Disable cuda')
        parser_group.add_argument('--volumeSaveFrequency', type=int, default=1,
                                  help="The frequency at which to save the volume")
        parser_group.add_argument('--onlyOpacityUntil', type=int, default=0,
                                  help="In epochs smaller than this value, optimize only the opacity, not the color")
        parser_group.add_argument('--memorySaving', action='store_true',
                                  help="Memory saving for large preshaded volumes.")

    def parse(self, parse_args):
        opt_dict:dict = vars(parse_args)
        self.opt_dict = opt_dict
        for (key, value) in opt_dict.items():
            setattr(self, key, value)
        return opt_dict

def train():
    # Settings
    parser = argparse.ArgumentParser(
        description='Volume function reconstruction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cfg = Config()
    cfg.init_parser(parser)
    opt = parser.parse_args()
    opt_dict = cfg.parse(opt)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(4)

    # settings
    s = Settings(cfg.settingsFile)
    reference_volume = s.load_dataset()
    if cfg.noCuda:
        reference_volume_data = reference_volume.getDataCpu(0)
    else:
        reference_volume.copy_to_gpu()
        reference_volume_data = reference_volume.getDataGpu(0)
    device = reference_volume_data.device
    rs = setup_default_settings(
        reference_volume, cfg.imgres, cfg.imgres,
        s.get_stepsize(), not cfg.noCuda)

    # multiscale
    if cfg.multiscale > 0:
        start_resolution = cfg.multiscale
        end_resolution = cfg.volres
        mipmap_levels = int(np.round(np.log2(end_resolution/start_resolution)))
        assert start_resolution*(2**mipmap_levels) == end_resolution,  \
            "the end resolution is not a power of the start resolution"
        print("start resolution:", start_resolution, ", end resolution:", end_resolution,
              "->", mipmap_levels, "mipmap levels")
        chunk_size = 32
    else:
        start_resolution = end_resolution = cfg.volres
        mipmap_levels = 0
        chunk_size = start_resolution

    # camera
    camera_config = s.get_camera()
    camera_pitch_cpu, camera_yaw_cpu = fibonacci_sphere(cfg.views, dtype=renderer_dtype_np)
    camera_distance_cpu = camera_config.distance * np.ones((cfg.views,), dtype=renderer_dtype_np)
    camera_center_cpu = np.stack([camera_config.center] * cfg.views, axis=0).astype(dtype=renderer_dtype_np)
    camera_fov_radians = camera_config.fov_y_radians
    camera_module = CameraOnASphere(camera_config.orientation)
    cameras = camera_module(
        torch.from_numpy(camera_center_cpu).to(device=device),
        torch.from_numpy(camera_yaw_cpu).to(device=device).unsqueeze(1),
        torch.from_numpy(camera_pitch_cpu).to(device=device).unsqueeze(1),
        torch.from_numpy(camera_distance_cpu).to(device=device).unsqueeze(1))

    # tf
    if cfg.tfmode == 'texture' or cfg.tfmode == 'linear':
        # always use texture
        cfg.tfmode = 'texture'
        cfg.opt_dict['tfmode'] = 'texture'
        rs.tf_mode = pyrenderer.TFMode.Texture
        _tf_points = s.get_tf_points()
        for p in _tf_points:
            p.val.w = max(p.val.w, cfg.minOpacity)
        tf_reference = pyrenderer.TFUtils.get_texture_tensor(_tf_points, 256)
        tf_reference = tf_reference.to(device=device, dtype=renderer_dtype_torch)
    elif cfg.tfmode == "identity":
        rs.tf_mode = pyrenderer.TFMode.Identity
        tf_reference = torch.tensor([[
            # r,g,b,a,pos
            [cfg.opacityScaling, 1]
        ]], dtype=renderer_dtype_torch, device=device)
    elif cfg.tfmode == "gauss":
        rs.tf_mode = pyrenderer.TFMode.Gaussian
        tf_reference = s.get_gaussian_tensor().to(device=device, dtype=renderer_dtype_torch)
    else:
        raise ValueError("unknown tfmode: " + cfg.tfmode)

    if cfg.randomizeTF:
        if cfg.preshaded:
            raise ValueError("randomized TFs and preshaded mode are incompatible")
        if cfg.batchMode != "stochastic":
            raise ValueError("randomized TFs only works with stochastic batch mode")
        print("Create randomized TFs")
        num_peaks = 3
        tf_cpu = np.zeros((cfg.views, num_peaks, 6), dtype=renderer_dtype_np)
        for v in range(cfg.views):
            R = np.random.randint(num_peaks)
            for r in range(0, R):
                tf_cpu[v][r][0] = np.random.rand()  # red
                tf_cpu[v][r][1] = np.random.rand()  # green
                tf_cpu[v][r][2] = np.random.rand()  # blue
                tf_cpu[v][r][3] = np.random.rand()*10+5  # opacity
                tf_cpu[v][r][4] = np.random.rand()*0.8+0.1  # mean
                tf_cpu[v][r][5] = (np.random.rand()**2) * 0.15 + 0.015  # variance
            for r in range(R, num_peaks):
                tf_cpu[v][r][5] = 1 # reset variance to avoid division by zero
        tf_random = torch.from_numpy(tf_cpu).to(device=device)

    memory_saving_mode = cfg.memorySaving
    if memory_saving_mode:
        if cfg.randomizeTF:
            raise ValueError("can't use memory saving option with randomized TFs")

    # volume
    if not cfg.preshaded:
        # regular volume
        volume_channels = 1
        volume_densities = VolumeDensities()
        if cfg.initial == "gauss":
            initial_volume_parameterized = cfg.initialGaussStd * torch.randn(
                (1, start_resolution, start_resolution, start_resolution),
                dtype=renderer_dtype_torch, device=device) + cfg.initialGaussMean
        elif cfg.initial == "sphere":
            initial_volume_class = pyrenderer.Volume.create_implicit(
                pyrenderer.ImplicitEquation.Sphere, start_resolution)
            if cfg.noCuda:
                initial_volume_parameterized = VolumeDensities.prepare_input(
                    torch.clamp(initial_volume_class.getDataCpu(0), 1e-3, 1-1e-3))
            else:
                initial_volume_class.copy_to_gpu()
                initial_volume_parameterized = VolumeDensities.prepare_input(
                    torch.clamp(initial_volume_class.getDataGpu(0), 1e-3, 1 - 1e-3))
        elif cfg.initial == "warp":
            # warp the reference volume by perlin noise
            from tests import generate_fractal_noise_3d
            options = dict([s.split('=') for s in cfg.warpOptions.split(';')])
            scale = float(options['scale'])
            res = int(options['res'])
            noise = generate_fractal_noise_3d(
                (start_resolution, start_resolution, start_resolution*3),
                (res, res, res),
                octaves = int(options.get('octaves', '4')),
                persistence=float(options.get('persistence', '0.5')),
                lacunarity=int(options.get('lacunarity', '2')),
            ) * scale
            noise = np.reshape(noise, (1, start_resolution, start_resolution, start_resolution, 3))
            X = torch.linspace(-1, 1, start_resolution)
            mx, my, mz = torch.meshgrid(X, X, X)
            grid = torch.stack((mz, my, mx), 3).unsqueeze(0)
            grid = grid + noise
            _initial_volume = F.grid_sample(
                F.interpolate(
                    reference_volume_data.unsqueeze(0), (start_resolution, start_resolution, start_resolution),
                    mode='trilinear', align_corners=True),
                grid.to(device=device, dtype=renderer_dtype_torch),
                mode='bilinear', padding_mode='border')[0]
            initial_volume_parameterized = VolumeDensities.prepare_input(
                torch.clamp(_initial_volume, 1e-4, 1 - 1e-3))
        elif cfg.initial == "file":
            path = cfg.initialFilePath
            epoch = cfg.initialFileEpoch
            if path is None or len(path)==0:
                raise ValueError("No path to an .hdf5 file with the initial volume specified")
            print("Load initial density volume from file", path)
            with h5py.File(path, 'r') as init_file:
                _volume_dset = init_file['volumes']
                if _volume_dset.shape[1] != 1:
                    raise ValueError("dset is not a density volume, expected shape B*1*X*Y*Z but got "+str(_volume_dset.shape))
                _initial_volume = _volume_dset[epoch, ...]
                _initial_volume = torch.from_numpy(_initial_volume).to(device=device, dtype=renderer_dtype_torch)
                if _initial_volume.shape[1] != start_resolution or \
                    _initial_volume.shape[2] != start_resolution or \
                    _initial_volume.shape[2] != start_resolution:
                    print("Initial density volume from file has the wrong resolution, resize it")
                    _initial_volume = F.interpolate(
                        _initial_volume.unsqueeze(0), (start_resolution, start_resolution, start_resolution),
                        mode='trilinear', align_corners=True)[0]
                initial_volume_parameterized = VolumeDensities.prepare_input(
                    torch.clamp(_initial_volume, 1e-4, 1 - 1e-3))
        else:
            raise ValueError("unknown volume initialization: " + cfg.initial)
    else:
        # pre-shaded volume
        volume_channels = 4
        volume_densities = VolumePreshaded()
        reference_volume_data = pyrenderer.TFUtils.preshade_volume(
            reference_volume_data, tf_reference, rs.tf_mode)
        rs.tf_mode = pyrenderer.TFMode.Preshaded
        rs.volume_filter_mode = pyrenderer.VolumeFilterMode.Preshaded

        assert cfg.initial == "gauss", "only gauss supported yet for preshading"
        initial_volume_parameterized = torch.randn(
            (4, start_resolution, start_resolution, start_resolution),
            dtype=renderer_dtype_torch, device=device)

    # renderer
    renderer = Renderer(rs, optimize_volume=True,
             gradient_method='adjoint')
    def render_images(renderer_instance, current_cameras,
                      volume_parameterized, tf = None, profiling=None):
        volume = volume_densities(volume_parameterized)
        if tf is None:
            tf = tf_reference
        images = renderer_instance(
            camera=current_cameras, fov_y_radians=camera_fov_radians,
            tf=tf_reference, volume=volume, profiling=profiling)
        return images, volume

    # reference
    print("Render reference images")
    reference_images = renderer(
            camera=cameras, fov_y_radians=camera_fov_radians,
            tf=tf_reference, volume=reference_volume_data)
    reference_images = toCHW(reference_images).detach()
    pyrenderer.sync()
    if memory_saving_mode:
        del reference_volume
        del reference_volume_data

    # define loss
    print("Define loss functions")
    class ImageLoss(torch.nn.Module):
        def __init__(self, cfg: Config):
            super().__init__()
            lb = LossBuilder(device)
            self._l1 = lb.l1_loss()
            self._l1_weight = cfg.l1
            self._l2 = lb.mse()
            self._l2_weight = cfg.l2
            self._dssim = lb.dssim_loss(4)
            self._dssim_weight = cfg.dssim
            self._lpips = lb.lpips_loss(4, 0, 1)
            self._lpips_weight = cfg.lpips
            if self._l1_weight==0 and self._l2_weight==0 and    \
                self._dssim_weight==0 and self._lpips_weight==0:
                raise ValueError("At least one image loss must be activated")

        def forward(self, img, reference, onlyOpacity):
            if onlyOpacity:
                img2 = img[:, 3:4, :, :]
                ref2 = reference[:, 3:4, :, :]
                l1 = self._l1(img2, ref2)
                l2 = self._l2(img2, ref2)
                img2 = torch.cat([img2]*4, dim=1)
                ref2 = torch.cat([ref2] * 4, dim=1)
                with dummy_context_mgr() if self._dssim_weight > 0 else torch.no_grad():
                    dssim = self._dssim(img2, ref2)
                with dummy_context_mgr() if self._lpips_weight > 0 else torch.no_grad():
                    lpips = self._lpips(img2, ref2)
            else:
                l1 = self._l1(img, reference)
                l2 = self._l2(img, reference)
                with dummy_context_mgr() if self._dssim_weight>0 else torch.no_grad():
                    dssim = self._dssim(img, reference)
                with dummy_context_mgr() if self._lpips_weight > 0 else torch.no_grad():
                    lpips = self._lpips(img, reference)
            loss = 0
            if self._l1_weight>0:
                loss = loss + self._l1_weight * l1
            if self._l2_weight>0:
                loss = loss + self._l2_weight * l2
            if self._dssim_weight>0:
                loss = loss + self._dssim_weight * dssim
            if self._lpips_weight>0:
                loss = loss + self._lpips_weight * lpips
            return loss, {
                'l1': l1.item(),
                'l2': l2.item(),
                'dssim': dssim.item(),
                'lpips': lpips.item()
            }

        def loss_names(self):
            return ["l1", "l2", "dssim", "lpips"]

    image_loss = ImageLoss(cfg)
    image_loss.to(device=device, dtype=renderer_dtype_torch)

    class VolumeLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._l2 = torch.nn.MSELoss()

        def forward(self, current_volume, reference_volume):
            l2 = self._l2(current_volume, reference_volume)
            return l2, {'l2vol': l2.item()}

        def loss_names(self):
            return ['l2vol']

    volume_loss = VolumeLoss()
    volume_loss.to(device=device, dtype=renderer_dtype_torch)

    class PriorLoss(torch.nn.Module):
        def __init__(self, cfg: Config):
            super().__init__()
            self._ps = SmoothnessPrior((1,2,3))
            self._ps_weight = cfg.priorSmoothing

        def forward(self, volume):
            loss = 0
            if self._ps_weight>0:
                ps = self._ps(volume)
                loss = loss + self._ps_weight * ps
            else:
                with torch.no_grad():
                    ps = self._ps(volume.detach())
            return loss, {
                'ps': ps.item()
            }

        def loss_names(self):
            return ["ps"]

    prior_loss = PriorLoss(cfg)
    prior_loss.to(device=device, dtype=renderer_dtype_torch)

    # optimization
    epochs = cfg.iterations
    epochs_with_volume = int(np.ceil(epochs/cfg.volumeSaveFrequency))
    epochs = cfg.volumeSaveFrequency * epochs_with_volume
    batch_size = cfg.batches
    num_batches = int(np.ceil(cfg.views / batch_size))
    optimizer_class = getattr(torch.optim, cfg.optimizer)
    optimizer_parameters = json.loads(cfg.optimParams)
    optimizer_parameters['lr'] = cfg.lr

    current_volume_parameterized = initial_volume_parameterized.clone()
    current_volume_parameterized.requires_grad_(True)
    optimizer = optimizer_class([current_volume_parameterized], **optimizer_parameters)

    # create output
    print("Create output file", cfg.outputFile)
    outputDir = os.path.split(cfg.outputFile)[0]
    os.makedirs(outputDir, exist_ok=True)
    with h5py.File(cfg.outputFile, 'w') as hdf5_file:
        for k,v in cfg.opt_dict.items():
            try:
                hdf5_file.attrs[k] = v
            except TypeError as ex:
                print("Exception", ex, "while saving attribute",k,"=",str(v))
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            hdf5_file.attrs['git'] = git_commit
            print("git commit", git_commit)
        except:
            print("unable to get git commit")
        hdf5_file.create_dataset(
            "reference_tf", data=tf_reference.cpu().numpy())
        hdf5_file.create_dataset(
            "initial_volume", data=volume_densities(initial_volume_parameterized).cpu().numpy())
        times = hdf5_file.create_dataset("times", (epochs+1,), dtype=renderer_dtype_np)
        volumes = hdf5_file.create_dataset(
            "volumes", ((mipmap_levels+1)*(epochs_with_volume+2), volume_channels, end_resolution, end_resolution, end_resolution),
            dtype=renderer_dtype_np, chunks=(1, volume_channels, chunk_size, chunk_size, chunk_size))
        volume_resolutions = hdf5_file.create_dataset(
            "volume_resolutions", ((mipmap_levels+1)*(epochs_with_volume+2), ),
            dtype=np.int32)
        losses = dict([
            (name, hdf5_file.create_dataset(name, ((mipmap_levels+1)*(epochs+1),), dtype=renderer_dtype_np))
            for name in image_loss.loss_names()+prior_loss.loss_names()+volume_loss.loss_names()+["total"]
        ])
        if cfg.randomizeTF:
            hdf5_file.create_dataset("random_tfs", data=tf_random.cpu().numpy())
        volume_iteration = 0
        loss_iteration = 0

        last_loss = None
        last_volume = None
        only_opacity = False

        # loop over mipmap-levels
        for mipmap in range(0, mipmap_levels+1):
            current_resolution = int(start_resolution*(2**mipmap))

            if not memory_saving_mode:
                reference_volume_resized = F.interpolate(
                    reference_volume_data.unsqueeze(0), (current_resolution, current_resolution, current_resolution),
                    mode='trilinear', align_corners=True)[0]

            if mipmap == 0:
                current_volume_parameterized = initial_volume_parameterized.clone()
            else:
                current_volume_parameterized = F.interpolate(
                    current_volume_parameterized.detach().unsqueeze(0), (current_resolution, current_resolution, current_resolution),
                    mode='trilinear', align_corners=True)[0].clone()
            current_volume_parameterized.requires_grad_(True)
            optimizer = optimizer_class([current_volume_parameterized], **optimizer_parameters)

            volumes[volume_iteration, :, :current_resolution, :current_resolution, :current_resolution] = \
                volume_densities(current_volume_parameterized).detach().cpu().numpy()
            volume_resolutions[volume_iteration] = current_volume_parameterized.shape[-1]
            volume_iteration += 1

            # optimize
            print("Now optimize with",num_batches,"batches for",epochs,"epochs at resolution",current_resolution)
            if cfg.batchMode == 'full':
                def optim_closure():
                    nonlocal last_loss, last_volume, only_opacity
                    optimizer.zero_grad()
                    total_loss = 0
                    partial_losses = defaultdict(float)
                    volume = None
                    for i in range(num_batches):
                        start = i*batch_size
                        end = min(cfg.views, (i+1)*batch_size)
                        images, volume = render_images(
                            renderer, cameras[start:end], current_volume_parameterized)
                        loss_value, lx = image_loss(
                            toCHW(images), reference_images[start:end], only_opacity)
                        for k,v in lx.items():
                            partial_losses[k] += v
                        loss_value = loss_value / num_batches
                        loss_value.backward()
                        total_loss += loss_value.item()
                    volume = volume_densities(current_volume_parameterized)
                    loss_value, lx = prior_loss(volume)
                    loss_value.backward()
                    for k, v in lx.items():
                        partial_losses[k] += v
                    total_loss += loss_value.item()
                    if not memory_saving_mode:
                        _, lx = volume_loss(volume, reference_volume_resized)
                        for k, v in lx.items():
                            partial_losses[k] += v
                    partial_losses['total'] = total_loss # .item()

                    last_volume = volume.detach()
                    last_loss = partial_losses
                    #total_loss.backward()
                    return total_loss

                start_time = time.time()
                with tqdm.tqdm(epochs+1) as iteration_bar:
                    for iteration in range(epochs+1):
                        only_opacity = iteration < cfg.onlyOpacityUntil
                        if cfg.onlyOpacityUntil>0 and iteration==cfg.onlyOpacityUntil:
                            # switching to full losses, reset optimizer (because of momentum)
                            optimizer = optimizer_class([current_volume_parameterized], **optimizer_parameters)
                        optimizer.step(optim_closure)

                        end_time = time.time()
                        times[iteration] = end_time-start_time
                        if iteration % cfg.volumeSaveFrequency == 0:
                            volumes[volume_iteration, :, :current_resolution, :current_resolution, :current_resolution] =\
                                last_volume.detach().cpu().numpy()
                            volume_resolutions[volume_iteration] = last_volume.shape[-1]
                            volume_iteration += 1
                        for k,v in last_loss.items():
                            losses[k][loss_iteration] = v / num_batches
                        loss_iteration += 1
                        iteration_bar.update(1)
                        iteration_bar.set_description("Loss: %7.5f" % last_loss['total'])
            elif cfg.batchMode == 'stochastic':
                view_indices = np.arange(cfg.views)
                tf_indices = np.arange(cfg.views)
                start_time = time.time()
                with tqdm.tqdm(epochs + 1) as iteration_bar:
                    for iteration in range(epochs + 1):
                        only_opacity = iteration < cfg.onlyOpacityUntil
                        if cfg.onlyOpacityUntil > 0 and iteration == cfg.onlyOpacityUntil:
                            # switching to full losses, reset optimizer (because of momentum)
                            optimizer = optimizer_class([current_volume_parameterized], **optimizer_parameters)

                        view_indices_permuted = np.random.permutation(view_indices)
                        view_indices_permuted = torch.from_numpy(view_indices_permuted).to(
                            device=device, dtype=torch.long)
                        if cfg.randomizeTF:
                            tf_indices_permuted = np.random.permutation(tf_indices)
                            tf_indices_permuted = torch.from_numpy(tf_indices_permuted).to(
                                device=device, dtype=torch.long)
                        partial_losses = defaultdict(float)
                        for i in range(num_batches):
                            start = i * batch_size
                            end = min(cfg.views, (i + 1) * batch_size)
                            current_views = view_indices_permuted[start:end]
                            with torch.no_grad():
                                if cfg.randomizeTF:
                                    current_tfs = tf_random[tf_indices_permuted[start:end]].detach()
                                    current_references = toCHW(renderer(
                                        camera=cameras[current_views], fov_y_radians=camera_fov_radians,
                                        tf=current_tfs, volume=reference_volume_data))
                                else:
                                    current_tfs = torch.cat([tf_reference]*(end-start), dim=0)
                                    current_references = reference_images[current_views]

                            def optim_closure():
                                nonlocal last_volume, only_opacity
                                optimizer.zero_grad()
                                total_loss = 0
                                images, volume = render_images(
                                    renderer, cameras[current_views],
                                    current_volume_parameterized, tf=current_tfs)
                                loss_value1, lx = image_loss(
                                    toCHW(images), current_references, only_opacity)
                                for k, v in lx.items():
                                    partial_losses[k] += v
                                loss_value2, lx = prior_loss(volume)
                                for k, v in lx.items():
                                    partial_losses[k] += v
                                loss_value = loss_value1 + loss_value2
                                loss_value.backward()
                                partial_losses['total'] += loss_value.item()
                                last_volume = volume.detach()
                                return loss_value
                            optimizer.step(optim_closure)

                        if not memory_saving_mode:
                            _, lx = volume_loss(last_volume, reference_volume_resized)
                            for k, v in lx.items():
                                partial_losses[k] += v

                        end_time = time.time()
                        times[iteration] = end_time - start_time
                        if iteration % cfg.volumeSaveFrequency == 0:
                            volumes[volume_iteration, :, :current_resolution, :current_resolution,
                            :current_resolution] = \
                                last_volume.detach().cpu().numpy()
                            volume_resolutions[volume_iteration] = last_volume.shape[-1]
                            volume_iteration += 1
                        for k, v in partial_losses.items():
                            losses[k][loss_iteration] = v / num_batches
                        loss_iteration += 1
                        iteration_bar.update(1)
                        iteration_bar.set_description("Loss: %7.5f" % partial_losses['total'])

    print("Done")


if __name__ == '__main__':
    train()
