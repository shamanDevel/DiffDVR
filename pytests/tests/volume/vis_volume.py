import os
import sys

sys.path.append(os.getcwd())

import h5py
import tests.vis_gui
import torch
import numpy as np
import skimage.transform

import pyrenderer

class UIVolume(tests.vis_gui.UI):

    def __init__(self, folder, preshaded_to_density):
        self._preshaded_to_density = preshaded_to_density
        if preshaded_to_density:
            keys = [
                "filename", "alpha", "beta", "initial", "seed"
            ]
            losses = ["data_loss", "prior_loss", "total"]
        else:
            keys = [
                "filename",
                #"reference", "tfmode",
                "preshaded",
                "views", "image_losses", "prior_losses", "min_op", "oou", "seed"
            ]
            losses = ["l1", "l2", "dssim", "lpips", "ps", 'l2vol']
        super().__init__(
            folder,
            keys,
            losses,
            512, 256,
            ["volumes", "volume_resolutions", "filename", "reference_tf", "preshaded"],
            delayed_loading=True,
            has_volume_slices=True)
        self.folder = folder

    def _createKey(self, hdf5_file: h5py.File):
        if self._preshaded_to_density:
            return self.Key(
                filename=os.path.splitext(os.path.split(hdf5_file.filename)[1])[0],
                alpha = "%.3f"%hdf5_file.attrs['alpha'],
                beta="%.3f" % hdf5_file.attrs['beta'],
                initial="%s" % hdf5_file.attrs['initial'],
                seed="%d" % hdf5_file.attrs['seed']
            )
        else:
            return self.Key(
                filename=os.path.splitext(os.path.split(hdf5_file.filename)[1])[0],
                #reference="%s" % os.path.splitext(os.path.split(hdf5_file.attrs['settingsFile'])[1])[0],
                #tfmode="%s" % (hdf5_file.attrs['tfmode'] if ('tfmode' in hdf5_file.attrs) else 'texture'),
                preshaded="1" if ('preshaded' in hdf5_file.attrs and hdf5_file.attrs['preshaded']) else "0",
                views="%d"%hdf5_file.attrs['views'],
                image_losses="l1=%.1f l2=%.1f dssim=%.1f lpips=%.1f" % (
                    hdf5_file.attrs['l1'], hdf5_file.attrs['l2'],
                    hdf5_file.attrs['dssim'], hdf5_file.attrs['lpips']
                ),
                prior_losses="%.4f"%hdf5_file.attrs['priorSmoothing'],
                min_op="%.4f" % hdf5_file.attrs['minOpacity'],
                oou="%d" % hdf5_file.attrs['onlyOpacityUntil'],
                seed="%d" % hdf5_file.attrs['seed']
                    )

    def _createValue(self, hdf5_file: h5py.File, filename: str):
        if self._preshaded_to_density:
            return self.Value(
                data_loss=hdf5_file['data_loss'][...],
                prior_loss=hdf5_file['prior_loss'][...],
                total=hdf5_file['total'][...],
                # volumes=hdf5_file['volumes'][...],
                volumes=hdf5_file['volumes'],
                volume_resolutions=hdf5_file['volume_resolutions'][...] if 'volume_resolutions' in hdf5_file else None,
                reference_tf=hdf5_file['reference_tf'][...],
                filename=os.path.splitext(filename)[0],
                preshaded='preshaded' in hdf5_file.attrs and hdf5_file.attrs['preshaded']
            )
        else:
            return self.Value(
                l1=hdf5_file['l1'][...],
                l2=hdf5_file['l2'][...],
                dssim=hdf5_file['dssim'][...],
                lpips=hdf5_file['lpips'][...],
                ps=hdf5_file['ps'][...],
                l2vol=hdf5_file['l2vol'][...],
                #volumes=hdf5_file['volumes'][...],
                volumes=hdf5_file['volumes'],
                volume_resolutions=hdf5_file['volume_resolutions'][...] if 'volume_resolutions' in hdf5_file else None,
                reference_tf=hdf5_file['reference_tf'][...],
                filename=os.path.splitext(filename)[0],
                preshaded='preshaded' in hdf5_file.attrs and hdf5_file.attrs['preshaded']
            )

    def get_num_epochs(self, current_value):
        return current_value.volumes.shape[0]

    def get_transfer_function(self, current_value, current_epoch):
        return self.tf_reference

    def _get_current_volume(self, current_value, current_epoch):
        if current_value.volume_resolutions is None:
            return current_value.volumes[current_epoch,...]
        else:
            res = current_value.volume_resolutions[current_epoch]
            return current_value.volumes[current_epoch, :, :res, :res, :res]

    def render_current_value(self, current_value, current_epoch):
        from diffdvr import renderer_dtype_torch
        if not current_value.preshaded:
            if len(current_value.volumes.shape)==5:
                # B * C=1 * X * Y * Z
                volume = self._get_current_volume(current_value, current_epoch)
            else:
                # old format, also no variable resolution
                # B * X * Y * Z
                volume = current_value.volumes[current_epoch:current_epoch + 1]
            print("Volume-Density min:", np.min(volume), ", max:", np.max(volume))
            volume = torch.from_numpy(volume).to(device=self.device, dtype=renderer_dtype_torch)
            tf = torch.from_numpy(self.tf_reference).to(device=self.device, dtype=renderer_dtype_torch)
            return self.renderer(camera=self.cameras, fov_y_radians=self.camera_fov_radians,
                                 tf=tf, volume=volume).detach().cpu().numpy()[0]
        else:
            volume = self._get_current_volume(current_value, current_epoch)
            if not volume.shape[0]==4:
                print("Expected four color channels for pre-shaded volumes, but got", volume.shape[0])
                return
            print("Volume-opacity min:", np.min(volume[3,...]), ", max:", np.max(volume[3,...]))
            volume = torch.from_numpy(volume).to(device=self.device, dtype=renderer_dtype_torch)
            tf = torch.from_numpy(self.tf_reference).to(device=self.device, dtype=renderer_dtype_torch)
            current_tf_mode = pyrenderer.TFMode(int(self.renderer.settings.tf_mode))
            current_filter_mode = pyrenderer.VolumeFilterMode(int(self.renderer.settings.volume_filter_mode))
            self.renderer.settings.tf_mode = pyrenderer.TFMode.Preshaded
            self.renderer.settings.volume_filter_mode = pyrenderer.VolumeFilterMode.Preshaded
            out = self.renderer(camera=self.cameras, fov_y_radians=self.camera_fov_radians,
                                 tf=tf, volume=volume).detach().cpu().numpy()[0]
            self.renderer.settings.tf_mode = current_tf_mode
            self.renderer.settings.volume_filter_mode = current_filter_mode
            return out

    def get_slice(self, is_reference: bool, current_value, current_epoch,
                  slice: float, axis : str):
        if is_reference:
            volume = self.volume.getDataCpu(0).numpy()
        else:
            if len(current_value.volumes.shape)==5:
                volume = self._get_current_volume(current_value, current_epoch)
            else:
                volume = current_value.volumes[current_epoch:current_epoch+1]

        if axis == 'x':
            slice_index = int(slice * (volume.shape[1]-1))
            slice = volume[:, slice_index, :, :]
        elif axis == 'y':
            slice_index = int(slice * (volume.shape[2]-1))
            slice = volume[:, :, slice_index, :]
        elif axis == 'z':
            slice_index = int(slice * (volume.shape[3]-1))
            slice = volume[:, :, :, slice_index]
        else:
            raise ValueError("Unknown slice axis: " + axis)

        if slice.shape[0] == 1:
            print("Slice-Density min:", np.min(slice), ", max:", np.max(slice),
                  "of reference" if is_reference else "")
            slice_rgb = np.stack([slice[0]] * 3, axis=2)
        else:
            print("Slice-Opacity min:", np.min(slice[3]), ", max:", np.max(slice[3]),
                  "of reference" if is_reference else "")
            slice_rgb = np.stack([slice[0], slice[1], slice[2]], axis=2)
            slice_rgb = slice_rgb * np.clip(slice[3,:,:,np.newaxis], None, 1) # opacity
        slice_rgb = skimage.transform.resize(slice_rgb, (self.ImgRes, self.ImgRes))
        return slice_rgb

if __name__ == "__main__":
    #basePath = os.path.join(os.getcwd(), "..\\..\\results")
    basePath = "D:\\DiffDVR-Results"

    #ui = UIVolume(os.path.join(basePath, "volume\\recon1"), False)
    #ui = UIVolume(os.path.join(basePath, "volume\\recon1"), True)

    #ui = UIVolume(os.path.join(basePath, "volume\\tooth2"), False)
    #ui = UIVolume(os.path.join(basePath, "volume\\tooth2"), True)

    #ui = UIVolume(os.path.join(basePath, "volume\\tooth2gauss"), False)
    #ui = UIVolume(os.path.join(basePath, "volume\\tooth2gauss"), True)

    #ui = UIVolume(os.path.join(basePath, "volume\\lobb1"), False)

    #ui = UIVolume(os.path.join(basePath, "volume\\skull4gauss"), False)
    #ui = UIVolume(os.path.join(basePath, "volume\\skull4gauss"), True)

    #ui = UIVolume(os.path.join(basePath, "volume\\skull5identity"), False)
    #ui = UIVolume(os.path.join(basePath, "volume\\skull6big"), False)

    #ui = UIVolume(os.path.join(basePath, "volume\\plume1"), False)
    ui = UIVolume(os.path.join(basePath, "volume\\big3"), False)
    #ui = UIVolume(os.path.join(basePath, "volume\\big3"), True)

    ui.show()
