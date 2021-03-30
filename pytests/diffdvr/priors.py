import torch
from typing import Union, Sequence

class SmoothnessPrior(torch.nn.Module):
  """
  n-dimensional smoothness prior loss.
  For each dimension i, the value $\int (f'_i(x))^2 dx$ is computed,
  i.e. the first derivative along dimension i, squared and summed/averaged over
  the image
  """

  def __init__(self, dim : Union[int, Sequence[int]], reduction : str = 'mean'):
    super().__init__()

    assert reduction in ["mean", "sum"]
    self._use_mean = reduction=='mean'

    if isinstance(dim, int):
      self._dims = [dim]
    elif isinstance(dim, tuple):
      self._dims = list(dim)
    else:
      raise ValueError("unknown type for 'dim', only int or tuple-of-ints supported")

  def forward(self, x):
    loss = 0
    for dim in self._dims:
      idx1 = (slice(None, None, None),) * dim + (slice(1, None, None),)
      idx2 = (slice(None, None, None),) * dim + (slice(None, -1, None),)
      dx = x[idx1] - x[idx2]
      dx2 = dx*dx
      if self._use_mean:
        loss = loss + torch.mean(dx2)
      else:
        loss = loss + torch.sum(dx2)
    return loss