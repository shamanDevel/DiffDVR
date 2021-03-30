"""
Export script for the reconstructions from
train_volume.py, which are normally visualized by vis_volume.py
"""

import os
import sys

sys.path.append(os.getcwd())

import h5py
import tests.vis_gui
import torch
import numpy as np
import skimage.transform
import imageio
from typing import List
from collections import namedtuple

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap

from diffdvr import Renderer, CameraOnASphere, Settings, setup_default_settings, \
    renderer_dtype_torch, toCHW
import pyrenderer
from losses.lossbuilder import LossBuilder

Columns = ["direct", "color", "density"]
Example = namedtuple("Example", ["name", "direct", "color", "density",
                                 "slice_pos", "slice_axis", "slice_rotation", "slice_crop", "slice_gamma"])

ImgRes = 512
SliceRes = 256
TfHeight = 128

def tf_to_texture(tf_mode: str, tf : np.ndarray, out: str):
    pixmap = QPixmap(ImgRes, TfHeight)
    tests.vis_gui.UI.VisualizeTF(tf_mode, tf, pixmap, background_color=(155,155,155))
    pixmap.save(out)

# Compute SSIM + PSNR
def blend_to_white(img_bhwc):
    img = torch.clone(img_bhwc)
    assert img.shape[3] == 4
    color = img[:,:,:,:3]
    alpha = img[:,:,:,3:]
    white = torch.ones_like(color)
    return alpha * color + (1-alpha) * white

class SSIM():
    def __init__(self):
        self._ssim = LossBuilder(torch.device("cpu")).ssim_loss(3)
    def __call__(self, x, y):
        return self._ssim(x, y).item()
ssim = SSIM()
class PSNR():
    def __call__(self, x, y):
        return 10 * torch.log10(1 / torch.nn.functional.mse_loss(
            x, y, reduction='mean')).item()
psnr = PSNR()


def get_slice(volume, slice: float, axis: str, rotation: int, crop: str, gamma: float):
    if axis == 'x':
        slice_index = int(slice * (volume.shape[1] - 1))
        slice = volume[:, slice_index, :, :]
    elif axis == 'y':
        slice_index = int(slice * (volume.shape[2] - 1))
        slice = volume[:, :, slice_index, :]
    elif axis == 'z':
        slice_index = int(slice * (volume.shape[3] - 1))
        slice = volume[:, :, :, slice_index]
    else:
        raise ValueError("Unknown slice axis: " + axis)

    if slice.shape[0] == 1:
        print("Slice-Density min:", np.min(slice), ", max:", np.max(slice))
        slice_rgb = np.stack([slice[0]] * 3, axis=2)
    else:
        print("Slice-Opacity min:", np.min(slice[3]), ", max:", np.max(slice[3]))
        slice_rgb = np.stack([slice[0], slice[1], slice[2]], axis=2)
        slice_rgb = slice_rgb * np.clip(slice[3, :, :, np.newaxis], None, 1)  # opacity
    slice_rgb = skimage.transform.resize(slice_rgb, (SliceRes, SliceRes))
    if rotation != 0:
        slice_rgb = np.rot90(slice_rgb, k=rotation)
    if len(crop)>0:
        x,y,w,h = map(int, crop.split(':'))
        slice_rgb = slice_rgb[y:y+h,x:x+w]
    if gamma != 1:
        slice_rgb = np.power(slice_rgb, 1/gamma)
    return slice_rgb

def vis(input_folder, output_folder, examples: List[Example]):
    os.makedirs(output_path, exist_ok=True)

    previous_settings_file = None
    # export images
    image_statistics = []
    renderer = Renderer()
    for example in examples:
        print("Export images for", example.name)
        image_statistics.append([])
        reference_image = None
        for column in Columns:
            filename = getattr(example, column)
            with h5py.File(os.path.join(input_folder, filename+".hdf5"), 'r') as hdf5_file:
                # settings
                settings_file = hdf5_file.attrs['settingsFile']
                tf_mode = hdf5_file.attrs['tfmode'] if 'tfmode' in hdf5_file.attrs else 'texture'
                tf_reference = hdf5_file["reference_tf"][...]
                volumes = hdf5_file['volumes']

                #prepare rendering
                if settings_file != previous_settings_file:
                    previous_settings_file = settings_file
                    full_settings_file = os.path.abspath("..\\..\\" + settings_file)
                    print("Load settings from", full_settings_file)
                    s = Settings(full_settings_file)
                    volume = s.load_dataset()
                    volume.copy_to_gpu()
                    volume_data = volume.getDataGpu(0)
                    device = volume_data.device
                    rs = setup_default_settings(
                        volume, ImgRes, ImgRes, s.get_stepsize(), True)
                    renderer.settings = rs

                    camera_config = s.get_camera()
                    camera_yaw = camera_config.yaw_radians * torch.ones((1, 1), dtype=renderer_dtype_torch)
                    camera_pitch = camera_config.pitch_radians * torch.ones((1, 1), dtype=renderer_dtype_torch)
                    camera_distance = camera_config.distance * torch.ones((1, 1), dtype=renderer_dtype_torch)
                    camera_center = torch.from_numpy(np.array([camera_config.center])).to(
                        dtype=renderer_dtype_torch)
                    camera_fov_radians = camera_config.fov_y_radians
                    camera_module = CameraOnASphere(camera_config.orientation)
                    cameras = camera_module(camera_center, camera_yaw, camera_pitch, camera_distance)
                    cameras = cameras.to(device=device)

                if tf_mode == 'texture':
                    rs.tf_mode = pyrenderer.TFMode.Texture
                elif tf_mode == 'linear':
                    rs.tf_mode = pyrenderer.TFMode.Linear
                elif tf_mode == "identity":
                    rs.tf_mode = pyrenderer.TFMode.Identity
                elif tf_mode == "gauss":
                    rs.tf_mode = pyrenderer.TFMode.Gaussian
                else:
                    raise ValueError("unknown tfmode: " + tf_mode)

                # render reference
                if reference_image is None:
                    tf_reference_torch = torch.from_numpy(tf_reference).to(
                        device=device, dtype=renderer_dtype_torch)
                    reference_image = blend_to_white(renderer(camera=cameras, fov_y_radians=camera_fov_radians,
                        tf=tf_reference_torch, volume=volume_data).detach())
                    imageio.imwrite(os.path.join(output_folder, "%s-reference-img.png"%example.name),
                                    reference_image.cpu().numpy()[0])
                    reference_image = toCHW(reference_image)
                    # render TF
                    tf_to_texture(tf_mode, tf_reference, os.path.join(output_folder, "%s-reference-tf.png"%example.name))
                    # export slice
                    imageio.imwrite(os.path.join(output_folder, "%s-reference-slice.png"%example.name),
                                    get_slice(volume_data.cpu().numpy(), example.slice_pos,
                                              example.slice_axis, example.slice_rotation, example.slice_crop, example.slice_gamma))

                # render current dataset
                if len(volumes.shape)==5:
                    current_volume = volumes[-1,...]
                else:
                    current_volume = volumes[-2:-1, ...]
                current_volume = torch.from_numpy(current_volume).to(device=device, dtype=renderer_dtype_torch)
                if current_volume.shape[0] == 1:
                    # density
                    current_image = blend_to_white(renderer(camera=cameras, fov_y_radians=camera_fov_radians,
                        tf=tf_reference_torch, volume=current_volume).detach())
                else:
                    # color
                    current_tf_mode = pyrenderer.TFMode(int(renderer.settings.tf_mode))
                    current_filter_mode = pyrenderer.VolumeFilterMode(int(renderer.settings.volume_filter_mode))
                    renderer.settings.tf_mode = pyrenderer.TFMode.Preshaded
                    renderer.settings.volume_filter_mode = pyrenderer.VolumeFilterMode.Preshaded
                    current_image = blend_to_white(renderer(camera=cameras, fov_y_radians=camera_fov_radians,
                                        tf=tf_reference_torch, volume=current_volume).detach())
                    renderer.settings.tf_mode = current_tf_mode
                    renderer.settings.volume_filter_mode = current_filter_mode

                imageio.imwrite(os.path.join(output_folder, "%s-%s-img.png" % (example.name, column)),
                                current_image.cpu().numpy()[0])
                current_image = toCHW(current_image)
                imageio.imwrite(os.path.join(output_folder, "%s-%s-slice.png" % (example.name, column)),
                                get_slice(current_volume.cpu().numpy(), example.slice_pos,
                                          example.slice_axis, example.slice_rotation, example.slice_crop, example.slice_gamma))

                # statistics
                image_statistics[-1].append((
                    psnr(reference_image, current_image),
                    ssim(reference_image, current_image)
                ))

    # write LaTeX
    with open(os.path.join(output_folder, "comparison.tex"), "w") as f:
        f.write("""
\\documentclass[border={-10pt 0pt 0pt 0pt}]{standalone}
\\usepackage{amsmath}
\\usepackage{graphicx}
\\usepackage{tikz}
\\usepackage{adjustbox}
\\usepackage{multirow}

\\begin{document}
	\\newlength{\\basewidth}
	\\setlength{\\basewidth}{30mm}
	\\setlength\\tabcolsep{1pt}
        """)

        # header
        f.write("\\begin{tabular}{c%s}\n" % ("@{\hskip 0.1in}".join(["cccc"]*len(examples))))
        for example in examples:
            f.write("& \multicolumn{4}{c}{%s}"%example.name)
        f.write("\\\\\n")
        for example in examples:
            f.write("& {\\small a) reference} & {\\small b) direct optim.} & {\\small c) color optim.} & {\\small d) density optim.}")
        f.write("\\\\\n")

        # renderings
        f.write("\\rotatebox[origin=c]{90}{Rendering}\n")
        for example in examples:
            for column in ["reference"]+Columns:
                f.write("& \\raisebox{-0.5\\height}{\\includegraphics[width=\\basewidth]{{%s-%s-img.png}}}\n"%(example.name, column))
        f.write("\\\\\n")

        # statistics
        for i,example in enumerate(examples):
            f.write("& \\multirow{2}{*}[0.0\\basewidth]{\\includegraphics[width=0.8\\basewidth]{{%s-reference-tf.png}}}"%example.name)
            for j,column in enumerate(Columns):
                f.write("& PSNR: $%.3f$dB"%image_statistics[i][j][0])
        f.write("\\\\\n")
        for i,example in enumerate(examples):
            f.write("&")
            for j,column in enumerate(Columns):
                f.write("& SSIM: $%.5f$"%image_statistics[i][j][1])
        f.write("\\\\[0.3em]\n")

        # slices
        f.write("\\rotatebox[origin=c]{90}{Slice}\n")
        for example in examples:
            for column in ["reference"] + Columns:
                f.write("& \\raisebox{-0.5\\height}{\\includegraphics[width=\\basewidth]{{%s-%s-slice.png}}}\n" % (
                example.name, column))

        f.write("""
    \\end{tabular}
\\end{document}
        """)

if __name__ == '__main__':
    a = QApplication([])
    path = "D:\\DiffDVR-Results\\volume\\big3"
    output_path = "D:\\DiffDVR-Results\\volume\\big3\\comparison"
    vis(path, output_path, [
        Example("Tooth", "tooth3gauss-direct-tiny", "tooth3gauss-pre-tiny", "tooth3gauss-recon1-small-ps200",
                0.48, "y", 1, "0:0:250:250", 1.0),
        Example("Thorax", "thorax2gauss256-direct-tiny", "thorax2gauss256-pre-small", "thorax2gauss256-recon-tiny",
                0.5, "z", 1, "", 2.0)
    ])
