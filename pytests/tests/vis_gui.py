import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
import os
import time
import h5py
import collections
import matplotlib
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# import PyQtChart
from PyQt5.QtChart import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from diffdvr import Renderer, CameraOnASphere, Settings, setup_default_settings, \
    renderer_dtype_torch
import pyrenderer

class UI(ABC):

    def __init__(
            self, folder,
            KeyNames, LossNames, ImgRes, TfHeight, ExtraValues = None,
            delayed_loading=False, has_volume_slices=False):
        self.folder = folder
        self.save_folder = os.path.split(folder)[0]
        self.delayed_loading = delayed_loading
        self.has_volume_slices = has_volume_slices

        self.KeyNames = KeyNames
        self.Key = collections.namedtuple("Key", KeyNames)
        self.LossNames = LossNames
        if ExtraValues is None:
            ExtraValues = ["tfs", "filename"]
        self.Value = collections.namedtuple("Value", LossNames + ExtraValues)
        self.ImgRes = ImgRes
        self.TFHeight = TfHeight
        self.ExportFileNames = "PNG (*.png);;JPEG (*.jpeg);;Bitmap (*.bmp)"

        self.renderer = Renderer()
        self.settings_file = None
        self.tf_reference = None
        self.tf_reference_torch = None
        self.tf_mode = None
        self.vis_mode = "image" # image, tf, slice
        self.slice_axis = 'x' # x,y,z
        self.bar_series = dict()
        self.white_background = False

        self.vis()

        self.reparse()

        self.img_reference_pixmap = None
        self.img_current_pixmap = None
        self.tf_reference_pixmap = None
        self.tf_current_pixmap = None
        self.slice_reference_pixmap = None
        self.slice_current_pixmap = None
        self.current_slice = 0
        self.lineseries_list = []

    def show(self):
        self.window.show()
        self.a.exec_()

    @abstractmethod
    def _createKey(self, hdf5_file: h5py.File):
        ...

    @abstractmethod
    def _createValue(self, hdf5_file: h5py.File, filename: str):
        ...

    def reparse(self):
        self.entries = self.parse(self.folder)
        self.entry_list = list(sorted(self.entries.items()))
        self.prepare_colormaps()
        self.current_value = None

        # fill data
        self.tableWidget.setRowCount(len(self.entries))
        for r, (k, v) in enumerate(self.entry_list):
            for c in range(len(self.KeyNames)):
                self.tableWidget.setItem(r, c, QTableWidgetItem(k[c]))
            for c in range(len(self.LossNames)):
                value = v[0][c][-1] if len(v[0][c]) > 0 else 1
                item = QTableWidgetItem("%.5f" % value)
                color = self.colorbar(self.normalizations[c](np.log(value)))
                item.setBackground(QColor(
                    int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                ))
                self.tableWidget.setItem(
                    r, c + len(self.KeyNames), item)

    def parse(self, folder):
        entries = dict()
        time_start = time.time()
        for filename in os.listdir(folder):
            if not filename.endswith(".hdf5"):
                continue
            hdf5_file = None
            try:
                hdf5_file = h5py.File(os.path.join(folder, filename), 'r')
                # get key
                if 'optimizer' in hdf5_file.attrs and hdf5_file.attrs['optimizer']=="Adamdelta":
                    continue
                key = self._createKey(hdf5_file)
                # get value
                value = self._createValue(hdf5_file, filename)
                # settings
                settings_file = hdf5_file.attrs['settingsFile']
                tf_mode = hdf5_file.attrs['tfmode'] if 'tfmode' in hdf5_file.attrs else 'texture'
                tf_reference = hdf5_file["reference_tf"][...] if "reference_tf" in hdf5_file else None
                # add entry
                entries[key] = (value, settings_file, tf_mode, tf_reference)
                if not self.delayed_loading:
                    hdf5_file.close()
                print("Loaded:", filename)
            except:
                print("Unable to load file", filename, ":", sys.exc_info()[0])
                if hdf5_file is not None:
                    hdf5_file.close()
        time_end = time.time()
        print("Folder parsed in", (time_end-time_start), "seconds with", len(entries), " entries")
        return entries

    def prepare_rendering(self, settings_file, tf_mode, tf_reference):
        if settings_file != self.settings_file:
            self.settings_file = settings_file
            full_settings_file = os.path.abspath("..\\..\\"+settings_file)
            print("Load settings from", full_settings_file)
            s = Settings(full_settings_file)
            self.volume = s.load_dataset()
            self.volume.copy_to_gpu()
            self.volume_data = self.volume.getDataGpu(0)
            self.device = self.volume_data.device
            rs = setup_default_settings(
                self.volume, self.ImgRes, self.ImgRes, s.get_stepsize(), True)
            self.renderer.settings = rs
            self.tf_mode = None

            camera_config = s.get_camera()
            camera_yaw = camera_config.yaw_radians * torch.ones((1, 1), dtype=renderer_dtype_torch)
            camera_pitch = camera_config.pitch_radians * torch.ones((1, 1), dtype=renderer_dtype_torch)
            camera_distance = camera_config.distance * torch.ones((1, 1), dtype=renderer_dtype_torch)
            camera_center = torch.from_numpy(np.array([camera_config.center])).to(dtype=renderer_dtype_torch)
            self.camera_fov_radians = camera_config.fov_y_radians
            camera_module = CameraOnASphere(camera_config.orientation)
            cameras = camera_module(camera_center, camera_yaw, camera_pitch, camera_distance)
            self.cameras = cameras.to(device=self.device)

            self.selected_epoch = -1

        if tf_mode != self.tf_mode:
            self.tf_mode = tf_mode
            rs = self.renderer.settings
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
            self.tf_reference = None

        if self.tf_reference is None or np.any(self.tf_reference != tf_reference):
            self.tf_reference = tf_reference
            if tf_reference is not None:
                rs = self.renderer.settings
                if tf_reference.shape[2] == 4:
                    rs.tf_mode = pyrenderer.TFMode.Texture
                elif tf_reference.shape[2] == 5:
                    rs.tf_mode = pyrenderer.TFMode.Linear
                elif tf_reference.shape[2] == 6:
                    rs.tf_mode = pyrenderer.TFMode.Gaussian
                self.tf_reference_torch = torch.from_numpy(self.tf_reference).to(device=self.device, dtype=renderer_dtype_torch)
            self.visualize_reference()


    def render_with_tf(self, tf):
        tf = torch.from_numpy(tf).to(device=self.device, dtype=renderer_dtype_torch)
        return self.renderer(camera=self.cameras, fov_y_radians=self.camera_fov_radians,
                        tf=tf, volume=self.volume_data).detach().cpu().numpy()[0]

    def to_pixmap(self, img : np.ndarray):
        h, w, c = img.shape
        if c>3:
            if self.white_background:
                rgb = img[:,:,:3]
                alpha = img[:,:,3:]
                white = np.ones_like(rgb)
                img = alpha*rgb + (1-alpha)*white
            else:
                img = img[:,:,:3]
        img = np.ascontiguousarray(np.clip(255*img, 0, 255).astype(np.uint8))
        bytesPerLine = 3 * w
        qImg = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return QPixmap(qImg)


    def prepare_colormaps(self):
        self.normalizations = []
        for i in range(len(self.LossNames)):
            data = [None]*len(self.entry_list)
            for j in range(len(self.entry_list)):
                if len(self.entry_list[j][1][0][i])>0:
                    data[j] = self.entry_list[j][1][0][i][-1]
                else:
                    data[j] = 1
            data = np.log(np.clip(data, 1e-5, None))
            self.normalizations.append(matplotlib.colors.Normalize(
                np.min(data), np.max(data)
            ))
            print("Loss", self.LossNames[i], "-> min:", np.min(data), ", max:", np.max(data))
        self.colorbar = plt.get_cmap("Reds").reversed()


    def visualize_reference(self):
        # render reference
        if self.volume_data.shape[0] == 1:
            print("Reference Volume-Density min:", torch.min(self.volume_data).item(), ", max:", torch.max(self.volume_data).item())
        else: # preshaded
            print("Reference Volume-Opacity min:", torch.min(self.volume_data[3]).item(), ", max:", torch.max(self.volume_data[3]).item())
        img_reference_data = self.render_with_tf(self.tf_reference)
        self.img_reference_pixmap = self.to_pixmap(img_reference_data)
        self.reference_label.setPixmap(self.img_reference_pixmap)
        self.tf_reference_pixmap = self.visualize_tf(self.tf_reference, QPixmap(self.ImgRes, self.TFHeight))


    def visualize_tf(self, tf: np.ndarray, pixmap: QPixmap):
        _, R, _ = tf.shape
        W, H = pixmap.width(), pixmap.height()
        painter = QPainter(pixmap)

        def lerp(a, b, x):
            return (1 - x) * a + x * b

        if self.tf_mode == "texture":
            interpX = np.linspace(0, 1, W, endpoint=True)
            interpXp = np.array([(i+0.5)/R for i in range(R)])
            rx = np.interp(interpX, interpXp, tf[0, :, 0])
            gx = np.interp(interpX, interpXp, tf[0, :, 1])
            bx = np.interp(interpX, interpXp, tf[0, :, 2])
            ox = tf[0, :, 3]
            oxX = [0] + list(interpXp) + [1]
            oxY = [ox[0]] + list(ox) + [ox[-1]]
            max_opacity = np.max(ox)

            for x in range(W):
                painter.fillRect(x, 0, 1, H, QBrush(QColor(int(rx[x]*255), int(gx[x]*255), int(bx[x]*255))))

            lower = int(0.8 * H)
            upper = int(0.05 * W)
            def transform(x, y):
                return int(x*W), int(lerp(lower, upper, y/max_opacity))
            p1 = QPainterPath()
            p1.moveTo(0, 0)
            for x,y in zip(oxX, oxY):
                p1.lineTo(*transform(x, y))
            p1.lineTo(W, 0)
            p1.lineTo(0, 0)
            painter.fillPath(p1, QColor(255, 255, 255))

            pen = painter.pen()
            painter.setPen(QPen(QBrush(QColor(0,0,0)), 2, Qt.DashLine))
            painter.drawLine(0, lower, W, lower)

            painter.setPen(QPen(QBrush(QColor(0, 0, 0)), 5, Qt.SolidLine))
            points = [QPoint(*transform(x,y)) for (x,y) in zip(oxX, oxY)]
            points = [[p1, p2] for (p1, p2) in zip(points[:-1], points[1:])]
            points = [p for p2 in points for p in p2 ]
            painter.drawLines(*points)
            painter.setBrush(QColor(0,0,0))
            for x, y in zip(oxX[1:-1], oxY[1:-1]):
                ix, iy = transform(x, y)
                painter.drawEllipse(int(ix-2), int(iy-2), 4, 4)
            painter.setPen(pen)

        elif self.tf_mode == "linear":
            pass

        elif self.tf_mode == "gauss":
            painter.fillRect(0, 0, W, H, QColor(50, 50, 50))
            # compute gaussians
            X = np.linspace(0, 1, W, endpoint=True)
            R = tf.shape[1]
            def normal(x, mean, variance):
                return np.exp(-(x-mean)*(x-mean)/(2*variance*variance))
            Yx = [None] * R
            max_opacity = 0
            for r in range(R):
                red, green, blue, opacity, mean, variance = tuple(tf[0,r,:])
                Yx[r] = normal(X, mean, variance)*opacity
                max_opacity = max(max_opacity, opacity)
            # draw background gaussians
            for r in range(R):
                red, green, blue, _, _, _ = tuple(tf[0,r,:])
                col = QColor(int(red*255), int(green*255), int(blue*255), 100)
                for x in range(W):
                    y = int(Yx[r][x] * H / max_opacity)
                    if y>2:
                        painter.fillRect(x, H-y, 1, y, col)
            # draw foreground gaussians
            for r in range(R):
                red, green, blue, _, _, _ = tuple(tf[0,r,:])
                col = QColor(int(red*255), int(green*255), int(blue*255), 255)
                for x in range(W):
                    y = int(Yx[r][x] * H / max_opacity)
                    if y>2:
                        painter.fillRect(x, H-y, 1, 2, col)


            # draw foreground gaussians

        del painter
        return pixmap

    def get_num_epochs(self, current_value):
        return current_value.tfs.shape[0]

    def selection_changed(self, row, column):
        print("Render image for row", row)
        self.current_value, settings_file, tf_mode, tf_reference = self.entry_list[row][1]
        self.prepare_rendering(settings_file, tf_mode, tf_reference)
        self.img_current_box.setTitle("Current: "+self.current_value.filename)
        num_epochs = self.get_num_epochs(self.current_value)
        self.epoch_slider.setMaximum(num_epochs)
        self.selected_epoch = min(self.selected_epoch, num_epochs-1)
        if self.selected_epoch == -1:
            self.selected_epoch = num_epochs-1
        self.epoch_slider.setValue(self.selected_epoch)
        self.epoch_slider_changed()
        self.slice_slider_changed()

        # CHARTS

        rows = [i.row() for i in self.tableWidget.selectedIndexes()]
        rows = sorted(set(rows))
        print("Show plots for rows", rows)
        # remove old rows
        indices_to_remove = set(self.bar_series.keys()) - set(rows)
        for i in indices_to_remove:
            self.series.remove(self.bar_series[i])
            del self.bar_series[i]
        # add new
        indices_to_add = set(rows) - set(self.bar_series.keys())
        for i in indices_to_add:
            value = self.entry_list[row][1][0]
            bset = QBarSet(value.filename)
            for l in range(len(self.LossNames)):
                bset.append(value[l][-1] if len(value[l]>0) else 1)
            self.bar_series[i] = bset
            self.series.append(bset)
        # adjust range
        max_loss = [
            (self.entry_list[r][1][0][l][-1] if len(self.entry_list[r][1][0][l])>0 else 1)
                for l in range(len(self.LossNames))
                for r in rows
        ]
        max_loss = np.max(max_loss)
        print("max loss:", max_loss)
        self.axisY.setRange(0, max_loss * 1.1)

        if len(rows)==1:
            # only one entry visible, switch to line chart
            for s in self.lineseries_list:
                self.linechart.removeSeries(s)
            self.lineseries_list.clear()
            value = self.entry_list[row][1][0]
            max_loss = np.max([
                np.max(value[idx])
                for idx in range(len(self.LossNames))
            ])
            min_loss = np.min([
                np.min(value[idx])
                for idx in range(len(self.LossNames))
            ])
            self.lineAxisX.setRange(0, len(value[0]))
            self.lineAxisY.setRange(min_loss * 0.97, max_loss * 1.03)
            for idx,name in enumerate(self.LossNames):
                s = QLineSeries()
                for x,y in enumerate(value[idx]):
                    s.append(x, y)
                #print(name, value[idx])
                s.setName(name)
                self.linechart.addSeries(s)
                s.attachAxis(self.lineAxisX)
                s.attachAxis(self.lineAxisY)
                self.lineseries_list.append(s)

            self.chart_stack.setCurrentIndex(1)
        else:
            self.chart_stack.setCurrentIndex(0)

    def render_current_value(self, current_value, current_epoch):
        tf = self.current_value.tfs[self.selected_epoch:self.selected_epoch + 1, :, :]
        img = self.render_with_tf(tf)
        return img

    def get_transfer_function(self, current_value, current_epoch):
        return self.current_value.tfs[self.selected_epoch:self.selected_epoch + 1, :, :]

    def epoch_slider_changed(self):
        if self.current_value is None: return
        num_epochs = self.get_num_epochs(self.current_value)
        self.selected_epoch = self.epoch_slider.value()
        self.selected_epoch = min(self.selected_epoch, num_epochs - 1)
        self.on_epoch_changed(self.current_value, self.selected_epoch)
        self.epoch_label.setText("%d"%self.selected_epoch)
        img = self.render_current_value(self.current_value, self.selected_epoch)
        self.img_current_pixmap = self.to_pixmap(img)
        tf = self.get_transfer_function(self.current_value, self.selected_epoch)
        self.tf_current_pixmap = self.visualize_tf(tf, QPixmap(self.ImgRes, self.TFHeight))
        if self.has_volume_slices:
            self.slice_current_pixmap = self.to_pixmap(self.get_slice(
                False, self.current_value, self.selected_epoch, self.current_slice, self.slice_axis))
        if self.vis_mode=='tf':
            self.current_label.setPixmap(self.tf_current_pixmap)
        elif self.vis_mode=='image':
            self.current_label.setPixmap(self.img_current_pixmap)
        elif self.vis_mode=='slices':
            self.current_label.setPixmap(self.slice_current_pixmap)

    def on_epoch_changed(self, current_value, current_epoch):
        pass # overwritten in subclasses

    def get_slice(self, is_reference: bool, current_value, current_epoch,
                  slice: float, axis: str):
        raise NotImplementedError("Must be implemented by subclasses supporting volume slices")

    def slice_slider_changed(self):
        if self.current_value is None: return
        if not self.has_volume_slices: return
        self.current_slice = self.slice_slider.value() / 100.0
        self.slice_reference_pixmap = self.to_pixmap(self.get_slice(
            True, self.current_value, 0, self.current_slice, self.slice_axis))
        self.slice_current_pixmap = self.to_pixmap(self.get_slice(
            False, self.current_value, self.selected_epoch, self.current_slice, self.slice_axis))
        if self.vis_mode == 'slices':
            self.reference_label.setPixmap(self.slice_reference_pixmap)
            self.current_label.setPixmap(self.slice_current_pixmap)

    def slice_axis_changed(self, axis):
        self.slice_axis = axis
        self.slice_slider_changed() # redraw

    def switch_vis_mode(self, vis_mode):
        self.vis_mode = vis_mode
        if self.vis_mode=='tf':
            if self.tf_reference_pixmap is not None:
                self.reference_label.setPixmap(self.tf_reference_pixmap)
            if self.current_value is not None:
                self.current_label.setPixmap(self.tf_current_pixmap)
        elif self.vis_mode=='image':
            if self.img_reference_pixmap is not None:
                self.reference_label.setPixmap(self.img_reference_pixmap)
            if self.current_value is not None:
                self.current_label.setPixmap(self.img_current_pixmap)
        elif self.vis_mode == 'slices':
            self.reference_label.setPixmap(self.slice_reference_pixmap)
            self.current_label.setPixmap(self.slice_current_pixmap)
        else:
            raise ValueError("Unknown vis mode: " + self.vis_mode)
        if self.slice_slider is not None:
            self.slice_slider.setEnabled(self.vis_mode == 'slices')

    def switch_background_mode(self, mode):
        if mode=="white":
            self.white_background = True
        else:
            self.white_background = False
        self.epoch_slider_changed()
        self.visualize_reference()

    def save_reference(self):
        print("Save reference")
        if self.vis_mode=='tf':
            pixmap = self.tf_reference_pixmap
            preferredFilename = "reference_tf.png"
        elif self.vis_mode=='image':
            pixmap = self.img_reference_pixmap
            preferredFilename = "reference_img.png"
        elif self.vis_mode=='slices':
            pixmap = self.slice_reference_pixmap
            preferredFilename = "reference_slice%02d.png"%int(self.current_slice*100)
        else:
            raise ValueError("Unknown vis mode: " + self.vis_mode)
        if pixmap is None:
            print("pixmap is none")
            return
        filename = QFileDialog.getSaveFileName(
            self.window, "Save reference rendering",
            os.path.join(self.save_folder, preferredFilename),
            self.ExportFileNames)[0]
        if filename is not None and len(filename)>0:
            pixmap.save(filename)
            print("Saved to", filename)

    def save_current(self):
        if self.current_value is None: return
        print("Save current")
        if self.vis_mode=='tf':
            pixmap = self.tf_current_pixmap
            preferredFilename = self.current_value.filename + "_epoch%03d_tf.png"%self.selected_epoch
        elif self.vis_mode=='image':
            pixmap = self.img_current_pixmap
            preferredFilename = self.current_value.filename + "_epoch%03d_img.png" % self.selected_epoch
        elif self.vis_mode=='slices':
            pixmap = self.slice_current_pixmap
            preferredFilename = self.current_value.filename + "_epoch%03d_slice%02d.png" % (
                self.selected_epoch, int(self.current_slice*100))
        else:
            raise ValueError("Unknown vis mode: " + self.vis_mode)
        if pixmap is None:
            print("pixmap is none")
            return
        filename = QFileDialog.getSaveFileName(
            self.window, "Save current rendering",
            os.path.join(self.save_folder, preferredFilename),
            self.ExportFileNames)[0]
        if filename is not None and len(filename) > 0:
            pixmap.save(filename)
            print("Saved to", filename)

    def create_browser(self, parent):
        parentLayout = QVBoxLayout(parent)
        self.reload_button = QPushButton("Reload", parent)
        self.reload_button.clicked.connect(lambda: self.reparse())
        parentLayout.addWidget(self.reload_button)
        self.tableWidget = QTableWidget(parent)
        self.tableWidget.setColumnCount(len(self.KeyNames)+len(self.LossNames))
        # header
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tableWidget.setHorizontalHeaderLabels(self.KeyNames+self.LossNames)
        # layout
        parentLayout.addWidget(self.tableWidget, stretch=1)
        parent.setLayout(parentLayout)
        # event
        self.tableWidget.cellClicked.connect(self.selection_changed)


    def create_charts(self, parent):
        parentLayout = QHBoxLayout(parent)
        self.chart_stack = QStackedWidget(parent)

        self.chart = QChart()
        self.series = QBarSeries()
        self.chart.addSeries(self.series)
        self.axisX = QBarCategoryAxis()
        self.axisX.append(self.LossNames)
        self.chart.addAxis(self.axisX, Qt.AlignBottom)
        self.series.attachAxis(self.axisX)
        self.axisY = QValueAxis()
        self.chart.addAxis(self.axisY, Qt.AlignLeft)
        self.series.attachAxis(self.axisY)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignRight)

        self.chartView = QChartView(self.chart, parent)
        self.chartView.setRenderHint(QPainter.Antialiasing)
        self.chart_stack.addWidget(self.chartView)

        self.linechart = QChart()
        self.lineAxisX = QValueAxis()
        self.lineAxisY = QLogValueAxis()
        #self.lineAxisY = QValueAxis()
        self.lineAxisX.setMinorGridLineVisible(True)
        self.lineAxisX.setMinorTickCount(10)
        self.linechart.addAxis(self.lineAxisX, Qt.AlignBottom)
        self.linechart.addAxis(self.lineAxisY, Qt.AlignLeft)
        self.linechart.legend().setVisible(True)
        self.linechart.legend().setAlignment(Qt.AlignRight)
        self.lineChartView = QChartView(self.linechart, parent)
        self.lineChartView.setRenderHint(QPainter.Antialiasing)
        self.chart_stack.addWidget(self.lineChartView)

        parentLayout.addWidget(self.chart_stack)

    def _custom_image_controls(self, parentLayout, parentWidget):
        pass

    def create_images(self, parent):
        parentLayout = QVBoxLayout(parent)

        layout1 = QHBoxLayout(parent)

        self.vis_mode_button_group = QButtonGroup(parent)
        self.radio_img = QRadioButton("Image", parent)
        if self.vis_mode=='image':
            self.radio_img.setChecked(True)
        layout1.addWidget(self.radio_img)
        self.vis_mode_button_group.addButton(self.radio_img)
        self.radio_tf = QRadioButton("TF", parent)
        if self.vis_mode=='tf':
            self.radio_tf.setChecked(True)
        layout1.addWidget(self.radio_tf)
        self.vis_mode_button_group.addButton(self.radio_tf)
        self.radio_img.clicked.connect(lambda: self.switch_vis_mode('image'))
        self.radio_tf.clicked.connect(lambda: self.switch_vis_mode('tf'))
        if self.has_volume_slices:
            self.radio_slices = QRadioButton("Slices", parent)
            if self.vis_mode == 'slices':
                self.radio_slices.setChecked(True)
            layout1.addWidget(self.radio_slices)
            self.vis_mode_button_group.addButton(self.radio_slices)
            self.radio_slices.clicked.connect(lambda: self.switch_vis_mode('slices'))

        self.background_button_group = QButtonGroup(parent)
        layout1.addWidget(QLabel("Background:"))
        self.background_white_button = QRadioButton("White", parent)
        if self.white_background:
            self.background_white_button.setChecked(True)
        layout1.addWidget(self.background_white_button)
        self.background_button_group.addButton(self.background_white_button)
        self.background_black_button = QRadioButton("Black", parent)
        if not self.white_background:
            self.background_black_button.setChecked(True)
        layout1.addWidget(self.background_black_button)
        self.background_button_group.addButton(self.background_black_button)
        self.background_white_button.clicked.connect(lambda: self.switch_background_mode('white'))
        self.background_black_button.clicked.connect(lambda: self.switch_background_mode('black'))

        layout1.addWidget(QLabel("Epoch:"))
        self.epoch_slider = QSlider(Qt.Horizontal, parent)
        self.epoch_slider.setMinimum(0)
        self.epoch_slider.setTracking(True)
        self.epoch_slider.valueChanged.connect(self.epoch_slider_changed)
        layout1.addWidget(self.epoch_slider)
        self.epoch_label = QLabel("0", parent)
        layout1.addWidget(self.epoch_label)

        parentLayout.addLayout(layout1)

        if self.has_volume_slices:
            layout3 = QHBoxLayout(parent)

            layout3.addWidget(QLabel("Axis:"))
            self.slice_axis_button_group = QButtonGroup(parent)

            self.radio_axis_x = QRadioButton("X", parent)
            if self.slice_axis == 'x':
                self.radio_axis_x.setChecked(True)
            layout3.addWidget(self.radio_axis_x)
            self.slice_axis_button_group.addButton(self.radio_axis_x)
            self.radio_axis_x.clicked.connect(lambda: self.slice_axis_changed('x'))

            self.radio_axis_y = QRadioButton("Y", parent)
            if self.slice_axis == 'y':
                self.radio_axis_y.setChecked(True)
            layout3.addWidget(self.radio_axis_y)
            self.slice_axis_button_group.addButton(self.radio_axis_y)
            self.radio_axis_y.clicked.connect(lambda: self.slice_axis_changed('y'))

            self.radio_axis_z = QRadioButton("Z", parent)
            if self.slice_axis == 'z':
                self.radio_axis_z.setChecked(True)
            layout3.addWidget(self.radio_axis_z)
            self.slice_axis_button_group.addButton(self.radio_axis_z)
            self.radio_axis_z.clicked.connect(lambda: self.slice_axis_changed('z'))

            layout3.addWidget(QLabel("Slice:"))
            self.slice_slider = QSlider(Qt.Horizontal, parent)
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(100)
            self.slice_slider.setTracking(True)
            self.slice_slider.valueChanged.connect(self.slice_slider_changed)
            layout3.addWidget(self.slice_slider)
            parentLayout.addLayout(layout3)
            if not self.vis_mode == 'slices':
                self.slice_slider.setEnabled(False)
        else:
            self.slice_slider = None

        self._custom_image_controls(parentLayout, parent)

        layout2 = QHBoxLayout(parent)
        layout2.addStretch(1)

        box1 = QGroupBox(parent)
        box1.setTitle("Reference")
        box1Layout = QHBoxLayout(box1)
        self.reference_label = QLabel(box1)
        self.reference_label.setFixedSize(self.ImgRes, self.ImgRes)
        self.reference_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.reference_label.customContextMenuRequested.connect(lambda e: self.save_reference())
        box1Layout.addWidget(self.reference_label)
        layout2.addWidget(box1)

        layout2.addStretch(1)

        box2 = QGroupBox(parent)
        box2.setTitle("Current")
        self.img_current_box = box2
        box2Layout = QHBoxLayout(box2)
        self.current_label = QLabel(box1)
        self.current_label.setFixedSize(self.ImgRes, self.ImgRes)
        self.current_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.current_label.customContextMenuRequested.connect(lambda e: self.save_current())
        box2Layout.addWidget(self.current_label)
        layout2.addWidget(box2)

        layout2.addStretch(1)
        parentLayout.addLayout(layout2, 1)
        parent.setLayout(parentLayout)


    def vis(self):
        self.a = QApplication([])
        self.window = QMainWindow()

        sizePolicyXY = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicyXY.setHorizontalStretch(0)
        sizePolicyXY.setVerticalStretch(0)
        sizePolicyX = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicyX.setHorizontalStretch(0)
        sizePolicyX.setVerticalStretch(0)
        sizePolicyY = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicyY.setHorizontalStretch(0)
        sizePolicyY.setVerticalStretch(0)

        self.window.setSizePolicy(sizePolicyXY)

        centralWidget = QWidget(self.window)
        gridLayout = QGridLayout(centralWidget)

        splitter2 = QSplitter(centralWidget)
        splitter2.setOrientation(Qt.Horizontal)
        splitter2.setChildrenCollapsible(False)
        splitter2.setSizePolicy(sizePolicyXY)

        browserBox = QGroupBox(splitter2)
        browserBox.setTitle("Parameters")
        browserBox.setSizePolicy(sizePolicyY)
        self.create_browser(browserBox)
        splitter2.addWidget(browserBox)

        splitter1 = QSplitter(splitter2)
        splitter1.setOrientation(Qt.Vertical)
        splitter1.setChildrenCollapsible(False)
        splitter1.setSizePolicy(sizePolicyXY)

        chartsBox = QGroupBox(splitter1)
        chartsBox.setTitle("Charts")
        chartsBox.setSizePolicy(sizePolicyY)
        self.create_charts(chartsBox)
        splitter1.addWidget(chartsBox)

        visBox = QGroupBox(splitter1)
        visBox.setTitle("Images")
        visBox.setSizePolicy(sizePolicyY)
        self.create_images(visBox)
        splitter1.addWidget(visBox)
        splitter2.addWidget(splitter1)
        splitter1.setStretchFactor(0, 1)
        splitter2.setStretchFactor(0, 1)

        gridLayout.addWidget(splitter2)
        self.window.setCentralWidget(centralWidget)
        self.window.resize(1024, 768)
        print("UI created")


if __name__ == "__main__":
    ui = UI(os.path.join(os.getcwd(), "..\\..\\results\\tf\\meta"))
    ui.show()
