import os
import sys

import h5py

sys.path.append(os.getcwd())

import os
import tests.vis_gui
import re

class UIStyletransfer(tests.vis_gui.UI):

    def __init__(self, folder, filter=".*"):
        self.filename_filer = re.compile(filter)
        super().__init__(
            folder,
            ["reference", "cropsize", "cropnumber", "views", "style_layers", "prior_losses", "seed"],
            ["style", "ps"],
            512, 256,
            ["tfs", "filename", "style_image"],
            delayed_loading=True)
        self.folder = folder

    def _createKey(self, hdf5_file: h5py.File):
        if self.filename_filer.fullmatch(hdf5_file.filename) is None:
            raise ValueError("file %s is ignored, does not match filename filter"%os.path.split(hdf5_file.filename)[1])
        return self.Key(
            reference="%s" % hdf5_file.attrs['reference'],
            cropsize="%d" % hdf5_file.attrs['cropsize'],
            cropnumber="%d" % hdf5_file.attrs['cropnumber'],
            views="%d"%hdf5_file.attrs['views'],
            style_layers="%s"%hdf5_file.attrs['styleLayers'],
            prior_losses="%.4f"%hdf5_file.attrs['priorSmoothing'],
            seed="%d" % hdf5_file.attrs['seed']
                )

    def _createValue(self, hdf5_file: h5py.File, filename: str):
        return self.Value(
                    style=hdf5_file['style'][...],
                    ps=hdf5_file['ps'][...],
                    tfs=hdf5_file['tfs'][...],
                    filename=os.path.splitext(filename)[0],
                    style_image=hdf5_file['style_image'][...]
                )

    def visualize_reference(self):
        self.img_reference_pixmap = None
        self.tf_reference_pixmap = None

    def selection_changed(self, row, column):
        super().selection_changed(row, column)
        style_image = self.current_value.style_image
        self.img_reference_pixmap = self.to_pixmap(style_image)
        self.reference_label.setPixmap(self.img_reference_pixmap)

if __name__ == "__main__":
    #ui = UIStyletransfer(os.path.join(os.getcwd(), "..\\..\\results\\tf\\style3"), filter=".*skull-refJet-cs64-cn4-v4-ps0050-conv_5x1,conv_7x1,conv_9x1.*")
    ui = UIStyletransfer(os.path.join(os.getcwd(), "..\\..\\results\\tf\\style3"))
    ui.show()
