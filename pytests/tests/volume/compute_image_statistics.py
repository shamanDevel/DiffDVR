import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
import imageio
from typing import List

from losses.lossbuilder import LossBuilder

def toCHW(bhwc : torch.Tensor):
  """
  Converts a tensor in BxHxWxC to BxCxHxW
  """
  return bhwc.movedim((0,1,2,3), (0,2,3,1))

class SSIM():
    def __init__(self):
        self._ssim = LossBuilder(torch.device("cpu")).ssim_loss(3)

    def __call__(self, x, y):
        return self._ssim(toCHW(torch.from_numpy(x).unsqueeze(0)),
                          toCHW(torch.from_numpy(y).unsqueeze(0))).item()

ssim = SSIM()


class PSNR():
    def __call__(self, x, y):
        return 10 * torch.log10(1 / torch.nn.functional.mse_loss(
            torch.from_numpy(x), torch.from_numpy(y), reduction='mean')).item()

psnr = PSNR()

def computeStatistics(path:str, reference:str, targets:List[str]):
    ref = imageio.imread(os.path.join(path,reference))[:,:,:3].astype(np.float32)/255
    print("%50s %8s %8s"%("Image", "PSNR", "SSIM"))
    for target in targets:
        t = imageio.imread(os.path.join(path,target))[:,:,:3].astype(np.float32)/255
        p = psnr(ref, t)
        s = ssim(ref, t)
        print("%50s %10s %10s"%(
            target,
            "%.3fdB"%p,
            "%.5f"%s
        ))

if __name__ == '__main__':
    path = "results/volume"

    computeStatistics(
        path,
        "tooth3gauss-reference_img.png",
        ["tooth3gauss-direct-tiny_epoch071_img.png",
         "tooth3gauss-pre-tiny_epoch071_img.png",
         "tooth3gauss-recon1-small-ps200_epoch011_img.png"]
    )

    computeStatistics(
        path,
        "thorax2gauss256-reference_img.png",
        ["thorax2gauss256-direct-tiny2_epoch071_img.png",
         "thorax2gauss256-pre-small_epoch071_img.png",
         "thorax2gauss256-recon-tiny_epoch011_img.png"]
    )