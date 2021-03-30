import os
import sys

sys.path.append(os.getcwd())

import h5py
import tests.vis_gui
import torch
import numpy as np
import skimage.transform
import matplotlib.colors
import matplotlib.pyplot

import pyrenderer

class UIStepsize(tests.vis_gui.UI):

    def __init__(self, folder):
        keys = [
            "filename", "mean", "lr"
        ]
        losses = ["min_stepsize", "max_stepsize"]
        extra_values = [
            "filename", "reference_tf", "stepsizes", "gradient_norms",
            "total_min_stepsize", "total_max_stepsize"]
        super().__init__(
            folder,
            keys,
            losses,
            512, 256,
            extra_values,
            delayed_loading=False,
            has_volume_slices=True)
        self.folder = folder

    def _createKey(self, hdf5_file: h5py.File):
        return self.Key(
            filename=os.path.splitext(os.path.split(hdf5_file.filename)[1])[0],
            mean="%.3f" % hdf5_file.attrs['meanStepsize'],
            lr="%.3f" % hdf5_file.attrs['lr'],
        )

    def _createValue(self, hdf5_file: h5py.File, filename: str):
        stepsizes = hdf5_file['stepsizes'][...]
        return self.Value(
            min_stepsize=hdf5_file['min_stepsizes'][...],
            max_stepsize=hdf5_file['max_stepsizes'][...],
            stepsizes=stepsizes,
            gradient_norms=hdf5_file['gradient_norm'][...],
            reference_tf=hdf5_file['reference_tf'][...],
            filename=os.path.splitext(filename)[0],
            total_min_stepsize=np.min(stepsizes),
            total_max_stepsize=np.max(stepsizes),
        )

    def get_num_epochs(self, current_value):
        return current_value.stepsizes.shape[0]

    def get_transfer_function(self, current_value, current_epoch):
        return self.tf_reference

    def render_current_value(self, current_value, current_epoch):
        from diffdvr import renderer_dtype_torch
        volume = self.volume_data
        stepsize = current_value.stepsizes[current_epoch]
        stepsize = torch.from_numpy(stepsize).to(device=self.device, dtype=renderer_dtype_torch)
        tf = torch.from_numpy(self.tf_reference).to(device=self.device, dtype=renderer_dtype_torch)

        inputs = self.renderer.settings.clone()
        inputs.camera_mode = pyrenderer.CameraMode.ReferenceFrame
        inputs.camera = pyrenderer.CameraReferenceFrame(self.cameras, self.camera_fov_radians)
        inputs.tf = tf
        inputs.volume = volume
        inputs.step_size = stepsize

        B = 1
        W = inputs.screen_size.x
        H = inputs.screen_size.y
        output_color = torch.empty(
            B, H, W, 4, dtype=volume.dtype, device=volume.device)
        output_termination_index = torch.empty(
            B, H, W, dtype=torch.int32, device=volume.device)
        outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)

        pyrenderer.Renderer.render_forward(inputs, outputs)

        return output_color.detach().cpu().numpy()[0]

    def get_slice(self, is_reference: bool, current_value, current_epoch,
                  slice: float, axis : str):
        if slice < 0.5:
            stepsize = current_value.stepsizes[current_epoch]
            norm = matplotlib.colors.LogNorm(
                vmin=current_value.total_min_stepsize,
                vmax=current_value.total_max_stepsize)
            cm = matplotlib.pyplot.get_cmap("viridis").reversed()
            colors = cm(norm(stepsize[0]))
        else:
            gradient_norm = current_value.gradient_norms[current_epoch]
            norm = matplotlib.colors.LogNorm(
                vmin=max(1e-4, np.min(gradient_norm)),
                vmax=max(1e-4, np.max(gradient_norm)))
            cm = matplotlib.pyplot.get_cmap("inferno")
            colors = cm(norm(gradient_norm[0]))
        return colors[:,:,:3]

if __name__ == "__main__":
    #ui = UIStepsize(os.path.join(os.getcwd(), "..\\..\\results\\stepsize\\skull4gauss"))
    ui = UIStepsize(os.path.join(os.getcwd(), "..\\..\\results\\stepsize\\thorax2gauss"))
    ui.show()
