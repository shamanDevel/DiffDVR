
import sys
import os
sys.path.append(os.getcwd())

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
        parser_group.add_argument('-v', '--views', default=1, type=int, help="Number of views")
        parser_group.add_argument('-r', '--imgres', default=512, type=int, help="Image resolution")
        parser_group.add_argument('-tf', '--tfmode', default='texture',
                                  choices=['identity', 'texture', 'linear', 'gauss'])
        parser_group.add_argument('--opacityScaling', default=3, type=float,
                                  help="Opacity scaling, only used for identity TF")
        # model
        parser_group = parser.add_argument_group("Model")
        parser_group.add_argument('-m', '--meanStepsize', default=1, type=float, help="Mean step size in voxels")
        parser_group.add_argument('--ignoreEmpty', action='store_true', help="Ignore empty pixels during mean calculation")
        parser_group.add_argument('-s', '--scale', default=1, type=float, help="Downscaling/smoothing factor of the stepsize")
        parser_group.add_argument('--onlyOpacity', action="store_true", help="Only consider opacity when computing the gradient norm")
        parser_group.add_argument('--blendWhite', action="store_true",
                                  help="Blend to white before computing the norm. Excludes --onlyOpacity")
        # training
        parser_group = parser.add_argument_group("Training")
        parser_group.add_argument('-lr', default=1.2, type=float, help="The learning rate > 1")
        parser_group.add_argument('-i', '--iterations', default=50, type=int,
                                  help="The number of iterations in the training")
        parser_group.add_argument('--noCuda', action='store_true', help='Disable cuda')
        parser_group.add_argument('--seed', type=int, default=124, help='random seed to use. Default=124')

    def parse(self, parse_args):
        opt_dict: dict = vars(parse_args)
        self.opt_dict = opt_dict
        for (key, value) in opt_dict.items():
            setattr(self, key, value)
        return opt_dict


def train():
    # Settings
    parser = argparse.ArgumentParser(
        description='Stepsize optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cfg = Config()
    cfg.init_parser(parser)
    opt = parser.parse_args()
    opt_dict = cfg.parse(opt)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(4)

    if cfg.onlyOpacity and cfg.blendWhite:
        raise ValueError("cannot specifiy both --onlyOpacity and --blendWhite at the same time")

    # settings
    s = Settings(cfg.settingsFile)
    volume = s.load_dataset()
    if cfg.noCuda:
        volume_data = volume.getDataCpu(0)
    else:
        volume.copy_to_gpu()
        volume_data = volume.getDataGpu(0)
    device = volume_data.device
    rs: pyrenderer.RendererInputs = setup_default_settings(
        volume, cfg.imgres, cfg.imgres,
        s.get_stepsize(), not cfg.noCuda)
    rs.volume = volume_data

    # camera
    camera_config = s.get_camera()
    if cfg.views == 1:
        # use test view
        camera_pitch_cpu = np.array([camera_config.pitch_radians], dtype=renderer_dtype_np)
        camera_yaw_cpu = np.array([camera_config.yaw_radians], dtype=renderer_dtype_np)
    else:
        # random views
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
    rs.camera_mode = pyrenderer.CameraMode.ReferenceFrame
    rs.camera = pyrenderer.CameraReferenceFrame(cameras, camera_fov_radians)

    # tf
    if cfg.tfmode == 'texture':
        rs.tf_mode = pyrenderer.TFMode.Texture
        _tf_points = s.get_tf_points()
        tf_reference = pyrenderer.TFUtils.get_texture_tensor(_tf_points, 256)
        tf_reference = tf_reference.to(device=device, dtype=renderer_dtype_torch)
    elif cfg.tfmode == 'linear':
        rs.tf_mode = pyrenderer.TFMode.Linear
        _tf_points = s.get_tf_points()
        tf_reference = pyrenderer.TFUtils.get_piecewise_tensor(_tf_points)
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
    rs.tf = tf_reference

    # stepsize
    ignore_empty = opt.ignoreEmpty
    volume_resolution = volume_data.shape[-1]
    mean_stepsize = cfg.meanStepsize / volume_resolution
    mean_cost = 1 / mean_stepsize
    initial_stepsize = mean_stepsize * torch.ones(cfg.views, cfg.imgres, cfg.imgres,
                                                  device=device, dtype=renderer_dtype_torch)
    lr = cfg.lr
    assert lr>1, "learning rate must be >1"

    # input / output structures
    fds = pyrenderer.ForwardDifferencesSettings()
    fds.D = 1
    fds.d_stepsize = 0

    output_color = torch.empty(
        cfg.views, cfg.imgres, cfg.imgres, 4, dtype=volume_data.dtype, device=volume_data.device)
    output_termination_index = torch.empty(
        cfg.views, cfg.imgres, cfg.imgres, dtype=torch.int32, device=volume_data.device)
    outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)

    forward_gradients_out = torch.zeros(cfg.views, cfg.imgres, cfg.imgres, fds.D, 4,
                                        dtype=volume_data.dtype, device=volume_data.device)

    # create output
    epochs = cfg.iterations
    print("Create output file", cfg.outputFile)
    outputDir = os.path.split(cfg.outputFile)[0]
    os.makedirs(outputDir, exist_ok=True)
    with h5py.File(cfg.outputFile, 'w') as hdf5_file:
        for k, v in cfg.opt_dict.items():
            try:
                hdf5_file.attrs[k] = v
            except TypeError as ex:
                print("Exception", ex, "while saving attribute", k, "=", str(v))
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            hdf5_file.attrs['git'] = git_commit
            print("git commit", git_commit)
        except:
            print("unable to get git commit")
        hdf5_file.create_dataset(
            "reference_tf", data=tf_reference.cpu().numpy())
        times = hdf5_file.create_dataset("times", (epochs,), dtype=renderer_dtype_np)
        min_stepsizes = hdf5_file.create_dataset("min_stepsizes", (epochs,), dtype=renderer_dtype_np)
        max_stepsizes = hdf5_file.create_dataset("max_stepsizes", (epochs,), dtype=renderer_dtype_np)
        stepsizes = hdf5_file.create_dataset("stepsizes", (epochs + 1,cfg.views, cfg.imgres, cfg.imgres), dtype=renderer_dtype_np)
        gradient_norms = hdf5_file.create_dataset("gradient_norm", (epochs + 1, cfg.views, cfg.imgres, cfg.imgres),
                                             dtype=renderer_dtype_np)
        stepsizes[0,...] = initial_stepsize.cpu().numpy()

        print("Optimize for", epochs, "iterations")
        current_stepsize = initial_stepsize.clone()
        start_time = time.time()
        #with tqdm.tqdm(epochs + 1) as iteration_bar:
        for iteration in range(epochs):
            # render with forward gradients
            rs.step_size = current_stepsize
            pyrenderer.Renderer.render_forward_gradients(
                rs, fds, outputs, forward_gradients_out)

            # in the first iteration, compute mask where the ray hit the bounding box
            if iteration==0:
                mask = output_termination_index>0
                fill_rate = torch.mean(mask*1.0).item()
                if cfg.scale > 1:
                    mask = torch.nn.functional.interpolate(mask.unsqueeze(0)*1.0, scale_factor=1 / cfg.scale, mode='bilinear')[0] > 0.5
                print("fill rate:", fill_rate)
            # compute gradient norm
            def nan_to_num(tensor):
                return tensor # not needed anymore, NaNs are fixed
                #return torch.where(torch.logical_or(torch.isnan(tensor), torch.isinf(tensor)), torch.zeros_like(tensor), tensor)

            forward_gradients_out2 = nan_to_num(forward_gradients_out[:,:,:,0,:])
            if cfg.onlyOpacity:
                gradient_norm = torch.abs(forward_gradients_out2[...,3])
            elif cfg.blendWhite:
                forward_gradients_out3 = forward_gradients_out2[...,:3]*output_color[...,3:] + (output_color[...,:3]-1)*forward_gradients_out2[...,3:]
                gradient_norm = torch.linalg.norm(forward_gradients_out3, dim=3)
            else:
                gradient_norm = torch.linalg.norm(forward_gradients_out2,dim=3)
            if cfg.scale>1:
                _, origSizeX, origSizeY = gradient_norm.shape
                gradient_norm = torch.nn.functional.interpolate(gradient_norm.unsqueeze(0), scale_factor=1/cfg.scale, mode='bilinear')[0]
                current_stepsize = torch.nn.functional.interpolate(current_stepsize.unsqueeze(0), scale_factor=1 / cfg.scale, mode='bilinear')[0]
            # per-pixel change
            per_pixel_change = current_stepsize * gradient_norm
            max_change = torch.max(per_pixel_change).item()
            # now decrease the stepsize where this value is high
            step_pre = current_stepsize / torch.pow(lr, per_pixel_change/max_change)
            # normalize
            #current_stepsize = step_pre * mean_stepsize / torch.mean(step_pre)
            cost_pre = 1 / step_pre
            if ignore_empty:
                cost_pre_filtered = torch.where(mask, cost_pre, mean_cost*torch.ones_like(cost_pre))
                #print("mean(cost_pre):", torch.mean(cost_pre).item(), ", mean(cost_pre_filtered):", torch.mean(cost_pre_filtered).item())
                current_stepsize = 1 / (cost_pre_filtered * mean_cost / torch.mean(cost_pre_filtered))
            else:
                current_stepsize = 1 / (cost_pre * mean_cost / torch.mean(cost_pre))
            if cfg.scale>1:
                current_stepsize = torch.nn.functional.interpolate(current_stepsize.unsqueeze(0), size=(origSizeX, origSizeY), mode='bilinear')[0]
                gradient_norm = torch.nn.functional.interpolate(gradient_norm.unsqueeze(0), size=(origSizeX, origSizeY), mode='bilinear')[0]
            # save
            min_stepsize = torch.min(current_stepsize).item()
            max_stepsize = torch.max(current_stepsize).item()
            stepsizes[iteration+1, ...] = current_stepsize.cpu().numpy()
            end_time = time.time()
            gradient_norms[iteration+1, ...] = gradient_norm.cpu().numpy()
            times[iteration] = end_time - start_time
            min_stepsizes[iteration] = min_stepsize
            max_stepsizes[iteration] = max_stepsize

            print("[Epoch %03d] change: %7.5f, min: %7.5f, max: %7.5f" % (
                iteration, max_change, min_stepsize*volume_resolution, max_stepsize*volume_resolution))
            if not np.isfinite(max_change):
                print("Inf or NaN!")
                break

                #iteration_bar.update(1)
                #iteration_bar.set_description("change: %7.5f, min: %7.5f, max: %7.5f" % (
                #    max_change, min_stepsize*volume_resolution, max_stepsize*volume_resolution))
            #break

    print("Done")


if __name__ == '__main__':
    train()
