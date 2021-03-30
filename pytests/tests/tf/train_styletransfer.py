"""
Large hyperparameter training session
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import os
import tqdm
import time
import h5py
import argparse
import json
from collections import defaultdict
import subprocess
import imageio

from diffdvr import Renderer, CameraOnASphere, Settings, setup_default_settings, \
    fibonacci_sphere, renderer_dtype_torch, renderer_dtype_np, ProfileRenderer, \
    TfTexture, SmoothnessPrior, toCHW
from losses import LossBuilder
import pyrenderer

class Config:
    def __init__(self):
        pass

    def init_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('outputFile', type=str, help="Output .hdf5 file")
        # dataset
        parser_group = parser.add_argument_group("Data")
        parser_group.add_argument('settingsFile', type=str, help="Settings .json file")
        parser_group.add_argument('reference', type=str, help="The reference file for the style transfer")
        parser_group.add_argument('-cs', '--cropsize', default=128, type=int, help="The size of the crops")
        parser_group.add_argument('-cn', '--cropnumber', default=4, type=int, help="The number of crops per image")
        parser_group.add_argument('-v', '--views', default=8, type=int, help="Number of views")
        parser_group.add_argument('-r', '--imgres', default=512, type=int, help="Image resolution")
        # model
        parser_group = parser.add_argument_group("Model")
        parser_group.add_argument('-tf', '--tfmode', default='texture',
                                  choices=['identity', 'texture', 'linear'])
        parser_group.add_argument('-R', '--tfres', default=64, type=int, help="TF resolution")
        parser_group.add_argument('--adjointImmediate', action='store_true',
                                  help="Use immediate adjoint mode instead of the (maybe faster) delayed mode")
        parser_group.add_argument('--minDensity', default=0.0, type=float, help="""
            The minimal density value in the TF that the optimizer can fill. 
            """)
        # losses
        parser_group = parser.add_argument_group("Loss")
        parser_group.add_argument('--styleLayers', default='conv_1,conv_3,conv_5', type=str, help="""
    Comma-separated list of layer names for the perceptual loss. 
    Note that the convolution layers are numbered sequentially: conv_1, conv2_, ... conv_19.
    Optinally, the weighting factor can be specified with a colon: "conv_4:1.0", if omitted, 1 is used.
    """)
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
        parser_group.add_argument('-b', '--batches', default=2, type=int,
                                  help="Batch size for training")
        parser_group.add_argument('--seed', type=int, default=124, help='random seed to use. Default=124')
        parser_group.add_argument('--noCuda', action='store_true', help='Disable cuda')

    def parse(self, parse_args):
        opt_dict:dict = vars(parse_args)
        self.opt_dict = opt_dict
        for (key, value) in opt_dict.items():
            setattr(self, key, value)
        return opt_dict

def train():
    # Settings
    parser = argparse.ArgumentParser(
        description='Transfer function reconstruction',
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
    volume = s.load_dataset()
    if cfg.noCuda:
        volume_data = volume.getDataCpu(0)
    else:
        volume.copy_to_gpu()
        volume_data = volume.getDataGpu(0)
    device = volume_data.device
    rs = setup_default_settings(
        volume, cfg.imgres, cfg.imgres,
        s.get_stepsize(), not cfg.noCuda)

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
    tfres = cfg.tfres
    min_density_in_pixels = int(cfg.minDensity * tfres)
    if cfg.tfmode == 'texture':
        rs.tf_mode = pyrenderer.TFMode.Texture
        _tf_points = s.get_tf_points()
        tf_reference = TfTexture.init_from_points(_tf_points, 256)
        tf_initial = torch.randn((1, tfres, 4), dtype=renderer_dtype_torch, device=device)
        tf_module = TfTexture()
    else:
        raise ValueError("Tf Mode '"+cfg.tfmode+"' not supported yet")

    # renderer
    renderer = Renderer(rs, optimize_tf=True,
             gradient_method='adjoint',
             tf_delayed_accumulation = not cfg.adjointImmediate)
    def render_images(renderer_instance,
                      current_cameras,
                      tf_parameterized,
                      profiling=None,
                      zero_padding=0):
        tf = tf_module(tf_parameterized)
        if zero_padding>0:
            B, R, C = tf.shape
            tf = torch.cat([
                torch.cat([
                    tf[:, :zero_padding, :-1],
                    torch.zeros(B, zero_padding, 1, device=tf.device, dtype=tf.dtype),
                ], dim=2),
                tf[:, zero_padding:, :]
            ], dim=1)
        images = renderer_instance(
            camera=current_cameras, fov_y_radians=camera_fov_radians,
            tf=tf, volume=volume_data, profiling=profiling)
        return images, tf

    # load reference image
    style_image_np = imageio.imread(cfg.reference)
    style_image = torch.from_numpy(style_image_np).to(dtype=renderer_dtype_torch, device=device) / 255
    print("The style image has a shape of", style_image.shape, "with maximal value", torch.max(style_image).item())
    assert style_image.shape[2] >= 3 # rgb(a)
    # create crops
    input_crops, style_crops = [], []
    def get_crop_for_shape(W, H, img = None, fill_rate = None):
        if img is None:
            x = np.random.randint(0, W - cfg.cropsize)
            y = np.random.randint(0, H - cfg.cropsize)
            return (y, y+cfg.cropsize, x, x+cfg.cropsize)
        else:
            trials = 50
            C = img.shape[2]
            for i in range(trials):
                y1,y2,x1,x2 = get_crop_for_shape(W, H)
                if C==4:
                    m = np.mean(style_image_np[y1:y2, x1:x2, 3])
                else:
                    m = np.mean(style_image_np[y1:y2, x1:x2])
                if m>=fill_rate:
                    return (y1,y2,x1,x2)
            print("Unable to find filled crop after iteration", trials)
            return get_crop_for_shape(W, H)
    for i in range(cfg.views):
        input_crops.append([get_crop_for_shape(cfg.imgres, cfg.imgres) for j in range(cfg.cropnumber)])
        style_crops.append([get_crop_for_shape(
            style_image.shape[1], style_image.shape[0], style_image_np, 0.5) for j in range(cfg.cropnumber)])
    # define loss
    class StyleLoss(torch.nn.Module):
        def __init__(self, style_image, input_crops, style_crops, style_layers):
            super().__init__()
            self.register_buffer('style_image', style_image)
            self.input_crops = input_crops
            self.style_crops = style_crops
            self.style_layers = [(s.split(':')[0],float(s.split(':')[1])) if ':' in s else (s,1) for s in style_layers.split(',')]
            lb = LossBuilder(device)
            self.pt_loss, self.style_losses, self.content_losses = \
                lb.get_style_and_content_loss(dict(), dict(self.style_layers))

        def forward(self, img, batch_offset: int):
            B, H, W, C = img.shape

            # build crops
            input_imgs = []
            style_imgs = []
            for b in range(B):
                icx = self.input_crops[b+batch_offset]
                scx = self.style_crops[b+batch_offset]
                for ic,sc in zip(icx, scx):
                    input_imgs.append(img[b, ic[0]:ic[1], ic[2]:ic[3], :3])
                    style_imgs.append(self.style_image[sc[0]:sc[1], sc[2]:sc[3], :3])
            style_img = torch.stack(style_imgs, dim=0)
            input_img = torch.stack(input_imgs, dim=0)
            combined_imgs = toCHW(torch.cat([style_img, input_img], dim=0))
            # eval losses
            style_score = torch.zeros((), dtype=img.dtype, device=img.device, requires_grad=True)
            self.pt_loss(combined_imgs)

            for sl in self.style_losses:
                style_score = style_score + sl.loss

            return style_score, {'style': style_score.item()}

        def loss_names(self):
            return ["style"]

    style_loss = StyleLoss(style_image, input_crops, style_crops, cfg.styleLayers)
    style_loss.to(device=device, dtype=renderer_dtype_torch)

    class PriorLoss(torch.nn.Module):
        def __init__(self, cfg: Config):
            super().__init__()
            self._ps = SmoothnessPrior(1)
            self._ps_weight = cfg.priorSmoothing

        def forward(self, tf):
            ps = self._ps(tf[:,:,:3])
            loss = 0
            if self._ps_weight>0:
                loss = loss + self._ps_weight * ps
            return loss, {
                'ps': ps.item()
            }

        def loss_names(self):
            return ["ps"]

    prior_loss = PriorLoss(cfg)
    prior_loss.to(device=device, dtype=renderer_dtype_torch)

    # optimization
    epochs = cfg.iterations
    batch_size = cfg.batches
    num_batches = int(np.ceil(cfg.views / batch_size))
    optimizer_class = getattr(torch.optim, cfg.optimizer)
    optimizer_parameters = json.loads(cfg.optimParams)
    optimizer_parameters['lr'] = cfg.lr

    current_tf_parameterized = tf_initial.clone()
    current_tf_parameterized.requires_grad_(True)
    optimizer = optimizer_class([current_tf_parameterized], **optimizer_parameters)

    # create output
    print("Create output file", cfg.outputFile)
    outputDir = os.path.split(cfg.outputFile)[0]
    os.makedirs(outputDir, exist_ok=True)
    with h5py.File(cfg.outputFile, 'w') as hdf5_file:
        for k,v in cfg.opt_dict.items():
            hdf5_file.attrs[k] = v
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            hdf5_file.attrs['git'] = git_commit
            print("git commit", git_commit)
        except:
            print("unable to get git commit")
        hdf5_file.create_dataset(
            "style_image", data=style_image.cpu().numpy())
        hdf5_file.create_dataset(
            "initial_tf", data=tf_module(tf_initial).cpu().numpy())
        times = hdf5_file.create_dataset("times", (epochs,), dtype=renderer_dtype_np)
        tfs = hdf5_file.create_dataset(
            "tfs", (epochs,tf_initial.shape[1], tf_initial.shape[2]),
            dtype=renderer_dtype_np)
        losses = dict([
            (name, hdf5_file.create_dataset(name, (epochs,), dtype=renderer_dtype_np))
            for name in style_loss.loss_names()+prior_loss.loss_names()+["total"]
        ])

        # optimize
        last_loss = None
        last_tf = None
        def optim_closure():
            nonlocal last_loss, last_tf
            optimizer.zero_grad()
            total_loss = 0
            partial_losses = defaultdict(float)
            for i in range(num_batches):
                start = i*batch_size
                end = min(cfg.views, (i+1)*batch_size)
                images, tf = render_images(
                    renderer, cameras[start:end], current_tf_parameterized,
                    zero_padding=min_density_in_pixels)
                loss_value, lx = style_loss(images, start)
                for k,v in lx.items():
                    partial_losses[k] += v
                total_loss += loss_value
            loss_value, lx = prior_loss(tf)
            for k, v in lx.items():
                partial_losses[k] += v
            total_loss += loss_value
            partial_losses['total'] = total_loss.item()

            last_tf = tf
            last_loss = partial_losses
            total_loss.backward()
            return total_loss

        start_time = time.time()
        with tqdm.tqdm(epochs) as iteration_bar:
            for iteration in range(epochs):
                optimizer.step(optim_closure)

                end_time = time.time()
                times[iteration] = end_time-start_time
                tfs[iteration] = last_tf.detach().cpu().numpy()[0]
                for k,v in last_loss.items():
                    losses[k][iteration] = v / num_batches
                iteration_bar.update(1)
                iteration_bar.set_description("Loss: %7.5f" % last_loss['total'])

    print("Done")


if __name__ == '__main__':
    train()
