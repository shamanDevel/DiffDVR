import os
import sys

import h5py

sys.path.append(os.getcwd())

import os
import tests.vis_gui

class UIMeta(tests.vis_gui.UI):

    def __init__(self, folder):
        super().__init__(
            folder,
            ["views", "image_losses", "prior_losses"],
            ["l1", "l2", "dssim", "lpips", "ps"],
            512, 256,
            ["tfs", "filename"])
        self.folder = folder

    def _createKey(self, hdf5_file: h5py.File):
        return self.Key(
                    views="%d"%hdf5_file.attrs['views'],
                    #optimizer=hdf5_file.attrs['optimizer'],
                    image_losses="l1=%.1f l2=%.1f dssim=%.1f lpips=%.1f"%(
                        hdf5_file.attrs['l1'], hdf5_file.attrs['l2'],
                        hdf5_file.attrs['dssim'], hdf5_file.attrs['lpips']
                    ),
                    prior_losses="%.4f"%hdf5_file.attrs['priorSmoothing']
                )

    def _createValue(self, hdf5_file: h5py.File, filename: str):
        return self.Value(
                    l1=hdf5_file['l1'][...],
                    l2=hdf5_file['l2'][...],
                    dssim=hdf5_file['dssim'][...],
                    lpips=hdf5_file['lpips'][...],
                    ps=hdf5_file['ps'][...],
                    tfs=hdf5_file['tfs'][...],
                    filename=os.path.splitext(filename)[0]
                )


if __name__ == "__main__":
    ui = UIMeta(os.path.join(os.getcwd(), "..\\..\\results\\tf\\meta"))
    ui.show()
