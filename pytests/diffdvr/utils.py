import atexit
import torch
import sys
import os
import numpy as np
from typing import Tuple, Union

try:
  import pyrenderer
except ModuleNotFoundError:
  __newpath = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../../bin'))
  sys.path.append(__newpath)
  print("Search pyrenderer in '%s'"%__newpath)
  import pyrenderer

@atexit.register
def __cleanup_renderer():
  pyrenderer.cleanup()

renderer_dtype_torch = torch.float64 if pyrenderer.use_double_precision() else torch.float32
renderer_dtype_np = np.float64 if pyrenderer.use_double_precision() else np.float32

def make_real3(vector):
  return pyrenderer.real3(vector[0], vector[1], vector[2])
def make_real4(vector):
  return pyrenderer.real4(vector[0], vector[1], vector[2], vector[3])

def cvector_to_numpy(vector : Union[pyrenderer.real3, pyrenderer.real4]):
  if isinstance(vector, pyrenderer.real3):
    return np.array([vector.x, vector.y, vector.z], dtype=renderer_dtype_np)
  elif isinstance(vector, pyrenderer.real4):
    return np.array([vector.x, vector.y, vector.z, vector.w], dtype=renderer_dtype_np)
  else:
    raise ValueError("unsupported type, real3 or real4 expected but got", type(vector))

def inverseSigmoid(y):
  """
  inverse of y=torch.sigmoid(y)
  :param y:
  :return: x
  """
  return torch.log(-y/(y-1))
class InverseSigmoid(torch.nn.Module):
  def forward(self, y):
    return inverseSigmoid(y)

def inverseSoftplus(y, beta=1, threshold=20):
  """
  inverse of y=torch.nn.functional.softplus(x, beta, threshold)
  :param y: the output of the softplus
  :param beta: the smoothness of the step
  :param threshold: the threshold after which a linear function is used
  :return: the input
  """
  return torch.where(y*beta>threshold, y, torch.log(torch.exp(beta*y)-1)/beta)
class InverseSoftplus(torch.nn.Module):
  def __init__(self, beta=1, threshold=20):
    super().__init__()
    self._beta = beta
    self._threshold = threshold
  def forward(self, y):
    return inverseSoftplus(y, self._beta, self._threshold)

def implies(x, y):
  """
  Returns "x implies y" / "x => y"
  """
  return not(x) or y

def toCHW(bhwc : torch.Tensor):
  """
  Converts a tensor in BxHxWxC to BxCxHxW
  """
  return bhwc.movedim((0,1,2,3), (0,2,3,1))

def fibonacci_sphere(N:int, *, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
  """
  Generates points on a sphere using the Fibonacci spiral
  :param N: the number of points
  :return: a tuple (pitch/latitude, yaw/longitude)
  """
  # Source: https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html
  gr = (np.sqrt(5.0)+1.0)/2.0 # golden ratio = 1.618...
  ga = (2-gr) * (2*np.pi)     # golden angle = 2.399...
  i = np.arange(1, N+1, dtype=dtype)
  lat = np.arcsin(-1 + 2*i/(N+1))
  lon = np.remainder(ga*i, 2*np.pi)
  #lon = np.arcsin(np.sin(ga*i))
  return lat, lon