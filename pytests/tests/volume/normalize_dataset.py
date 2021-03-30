"""
Normalizes a volume from [minDensity,maxDensity] to [0,1]
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import argparse
import numpy as np
import torch.nn.functional as F
import torch

from diffdvr import Renderer, CameraOnASphere, Settings, setup_default_settings, \
    renderer_dtype_torch, renderer_dtype_np
import pyrenderer

def normalize(settings_file, output_file, cubify=False, cubify_mode='pad'):
    print("Load settings", settings_file)
    s = Settings(settings_file)
    reference_volume = s.load_dataset()
    min_density = s._data["tfEditor"]["minDensity"]
    max_density = s._data["tfEditor"]["maxDensity"]
    print("min and max density:", min_density, max_density)
    data = reference_volume.getDataCpu(0)
    data = (data-min_density)/(max_density-min_density)
    data = torch.clamp(data, 0.0, 1.0)
    if not cubify:
        volume2 = pyrenderer.Volume.from_numpy(data[0].numpy())
        volume2.set_world_size(
            reference_volume.world_size.x,
            reference_volume.world_size.y,
            reference_volume.world_size.z)
    else:
        world_size = np.array([reference_volume.world_size.x,
            reference_volume.world_size.y,
            reference_volume.world_size.z])
        voxel_size = world_size / data.shape[1:]
        print("Old world size:", world_size)
        print("Voxel size:", voxel_size)
        # cubify
        def shift_bit_length(x):
            return 1 << (x - 1).bit_length()
        if cubify_mode == 'pad':
            def pad(a, dim):
                s = a.shape[dim]
                ns = shift_bit_length(s)
                if ns > s:
                    pad1 = (ns-s) // 2
                    pad0 = ns-s-pad1
                    pad_width = [(0,0) for i in range(dim)] + [(pad0, pad1)] + [(0,0) for i in range(dim+1, len(a.shape))]
                    return np.pad(a, pad_width, mode='constant')
                else:
                    return a

            data = data.numpy()
            data = pad(data, 1)
            data = pad(data, 2)
            data = pad(data, 3)
        elif cubify_mode == 'scale':
            data = F.interpolate(data.unsqueeze(1), size=(
                shift_bit_length(data.shape[1]), shift_bit_length(data.shape[2]), shift_bit_length(data.shape[3])
            ), mode='trilinear', align_corners=True).numpy()[:,0,...]
        new_world_size = voxel_size * data.shape[1:]
        new_world_size /= np.max(new_world_size)
        print("New world size:", new_world_size)
        volume2 = pyrenderer.Volume.from_numpy(data[0])
        volume2.set_world_size(
            new_world_size[0],
            new_world_size[1],
            new_world_size[2])

    print("Save to", output_file)
    volume2.save(output_file)

def main():
    parser = argparse.ArgumentParser(
        description='Volume normalization')
    parser.add_argument('input', help="Input setting file")
    parser.add_argument('output', help="Output volume file")
    parser.add_argument('--cubify', action='store_true')
    parser.add_argument('--cubify_mode', choices=['pad', 'scale'], default='pad')
    opt = parser.parse_args()
    normalize(opt.input, opt.output, opt.cubify, opt.cubify_mode)

if __name__ == '__main__':
    main()