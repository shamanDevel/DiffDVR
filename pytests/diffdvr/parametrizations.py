import torch
import torch.nn.functional as F
import numpy as np
from typing import Sequence

from diffdvr.utils import inverseSigmoid, inverseSoftplus
import pyrenderer

class VolumeDensities(torch.nn.Module):
    """
    Default parametrization of the density volume:
    The input which is optimized for is in the full range of [-infty,+infty].
    To convert it to valid densities in [0,1], it passed through a sigmoid function.
    """

    @staticmethod
    def prepare_input(original_volume):
        """
        Prepares the input for optimization: performs the inverse sigmoid on
        the original volume to get the input for optimization
        :param original_volume: the original volume densities in [0,1]
        :return: the transformed densities by an inverse sigmoid
        """
        return inverseSigmoid(original_volume)

    def forward(self, input):
        volume = torch.sigmoid(input)
        return volume

class VolumePreshaded(torch.nn.Module):
    """
    Default parametrization of the preshaded volume:
    The input which is optimized for is in the full range of [-infty,+infty].
    To convert it to valid densities in [0,1], it passed through a sigmoid function.
    """

    def forward(self, input):
        rgb = input[:3, :, :, :]
        opacity = input[3:, :, :, :]
        return torch.cat([
            torch.sigmoid(rgb),
            torch.nn.functional.softplus(opacity)
        ], dim=0)

class CameraOnASphere(torch.nn.Module):
    """
    Parametrization of a camera on a bounding sphere.
    Parameters:
      - orientation (pyrenderer.Orientation) fixed in the constructor
      - center (B*3) vector, specified in self.forward()
      - yaw (B*1) in radians, specified in self.forward()
      - pitch (B*1) in radians, specified in self.forward()
      - distance (B*1) scalar, specified in self.forward()
    """
    ZoomBase = 1.1

    def __init__(self, orientation : pyrenderer.Orientation):
        super().__init__()
        self._orientation = orientation

    def forward(self, center, yaw, pitch, distance):
        return pyrenderer.Camera.viewport_from_sphere(
            center, yaw, pitch, distance, self._orientation)

    @staticmethod
    def random_points(N : int, min_zoom = 6, max_zoom = 7, center = None):
        """
        Generates N random points on the sphere
        :param N: the number of points
        :param min_zoom: the minimal zoom factor
        :param max_zoom: the maximal zoom factor
        :param center: the center of the sphere. If None, [0,0,0] is used
        :return: a tuple with tensors (center, yaw, pitch, distance)
        """

        if center is None:
            center = [0,0,0]

        # random points on the unit sphere
        vec = np.random.randn(N, 3)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        # convert to pitch and yaw
        pitch = np.arccos(vec[:,2])
        yaw = np.arctan2(vec[:,1], vec[:,0])

        # sample distances
        dist = np.random.uniform(min_zoom, max_zoom, (N,))
        dist = np.power(CameraOnASphere.ZoomBase, dist)

        # to pytorch tensors
        dtype = torch.float64 if pyrenderer.use_double_precision() else torch.float32
        center = torch.tensor([center]*N, dtype=dtype)
        yaw = torch.from_numpy(yaw).to(dtype=dtype).unsqueeze(1)
        pitch = torch.from_numpy(pitch).to(dtype=dtype).unsqueeze(1)
        dist = torch.from_numpy(dist).to(dtype=dtype).unsqueeze(1)

        return center, yaw, pitch, dist

class TfPiecewiseLinear(torch.nn.Module):
    """
    Default parametrization for a piecewise linear transfer function:
    - Sigmoid for the color
    - Softplus for the opacity
    - ?? for the position

    It also contains factory functions to create a transfer function
    from a list of peaks
    """

    @staticmethod
    def create_from_peaks(peaks : np.ndarray, opacity_scaling = 1.0) -> torch.Tensor:
        """
        Creates a transfer function from a list of peaks
        :param peaks: an array of shape N*6 that describe the N peaks:
          - peaks[i,0:3] is the rgb color of the i-th peak
          - peaks[i,3] is the opacity of the i-th peak
          - peaks[i,4] is the position of the i-th peak
          - peaks[i,5] is the width of the i-th peak
        :param opacity_scaling: additional global scaling to the opacity
        :return: the transfer function tensor of shape 1*R*C
        """

        # first sort the peaks
        N = peaks.shape[0]
        assert peaks.shape[1] == 6
        peak_list = [(peaks[i,4], list(peaks[i,0:4]), peaks[i,5]) for i in range(N)]
        peak_list.sort()

        # build tf
        tf = []
        for i in range(N):
            pos, color, width = peak_list[i]
            # bounds for the peak
            if i==0:
                min_pos = 0
            else:
                min_pos = tf[-1][4]
            if i == N-1:
                max_pos = 1
            else:
                pos_next,_, width_next = peak_list[i+1]
                max_pos = max(0.5 * (pos+pos_next), pos_next-width_next)
            min_pos = max(min_pos, pos - width)
            max_pos = min(max_pos, pos + width)
            # add to tf
            tf.append(np.array(color[:3] + [0, min_pos]))
            tf.append(np.array(color[:3] + [color[3] * opacity_scaling, pos]))
            tf.append(np.array(color[:3] + [0, max_pos]))

        # add points at d=0 and d=1
        if tf[0][4] > 0:
            tf = [np.array([0,0,0,0,0])] + tf
        if tf[-1][4] < 1:
            tf.append(np.array([0,0,0,0,1]))
        tf = np.array(tf)
        dtype = torch.float64 if pyrenderer.use_double_precision() else torch.float32
        return torch.from_numpy(tf).to(dtype=dtype).unsqueeze(0)


class TfTexture(torch.nn.Module):
    """
    RGB: Sigmoid
    Alpha: Softplus
    """

    _softplus_beta = 1
    _softplus_threshold = 20

    def forward(self, tf_parameterized):
        rgb_pre = tf_parameterized[:,:,:3]
        alpha_pre = tf_parameterized[:,:,3:]
        rgb = torch.sigmoid(rgb_pre)
        alpha = F.softplus(
            alpha_pre, TfTexture._softplus_beta, TfTexture._softplus_threshold)
        return torch.cat([rgb, alpha], dim=2)

    @staticmethod
    def init_from_points(tf_points: Sequence[pyrenderer.TFPoint], R: int):
        """
        Initializes the optimized tensor from the list of TF points from the settings.
        You still have to cast the resulting tensor to the correct device and type
        :param tf_points:
        :param R: the resolution of the TF texture
        :return: the parameterized tensor to be optimized (inverse sigmoid, softplus)
        """
        tf = pyrenderer.TFUtils.get_texture_tensor(tf_points, R)
        rgb = tf[:,:,:3]
        alpha = tf[:,:,3:]
        pre_rgb = inverseSigmoid(rgb)
        pre_alpha = inverseSoftplus(
            alpha, TfTexture._softplus_beta, TfTexture._softplus_threshold)
        return torch.cat([pre_rgb, pre_alpha], dim=2)
