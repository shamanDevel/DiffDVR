import os
from os.path import join, dirname, realpath
import pytest
import time

import enoki as ek
import mitsuba
from mitsuba.core import Bitmap, Struct, Thread, EColorMode
from mitsuba.optix import OptiXRenderer
from mitsuba.render.autodiff import get_differentiable_parameters, Adam, SGD
import numpy as np

from render_helpers import load_scene, l2_loss, save_info_csv, integrator_type_for_scene, \
                           reference_differentiable_render, compute_image_gradients, \
                           ensure_valid_values
from config import CONFIGS, OUTPUT_DIR

DATA_KEY = '/Scene/Medium[id="medium1"]/Texture3D[id="medium-density"]/data'

def volume_to_torch(v):
    #volume_shape = (112,222,111) # z,y,z  (hard-coded)
    volume_shape = (256, 256, 256)
    v = v.reshape(*volume_shape)
    return v

def get_volume_from_params(params):
    return volume_to_torch(params[DATA_KEY].numpy())

def save_volume_slices(zyx, file_template):
    s = zyx.shape
    Bitmap(zyx[s[0]//2,:,:][...,np.newaxis].astype(np.float32)).write(file_template%'z')
    Bitmap(zyx[:,s[1]//2,:][...,np.newaxis].astype(np.float32)).write(file_template%'y')
    Bitmap(zyx[:,:,s[2]//2][...,np.newaxis].astype(np.float32)).write(file_template%'x')
    print("Volume slices saved to", file_template)

def load_or_render_ref(config):
    num_cameras = config['num_cameras']
    ref_fname = config['ref']
    ref_spp = config.get('ref_spp', 512)
    print('[ ] Rendering reference image with OptiX on the GPU...')
    ref_scene = config.get('ref_scene', config['scene'])
    scene = load_scene(ref_scene, parameters=config['ref_scene_params'])
    params = get_differentiable_parameters(scene)
    #save slices
    data = get_volume_from_params(params)
    save_volume_slices(data, (ref_fname%0)+".slice-%s.exr")
    #render
    renderer = OptiXRenderer(scene, integrator_type_for_scene(scene, with_gradients=False))
    for c in range(num_cameras):
        if os.path.isfile(ref_fname%c): continue
        scene.set_current_sensor(c)
        # current_b = renderer.render(seed=1234, spp=1024)
        current_b = renderer.render(seed=1234, spp=ref_spp, pass_count=8)
        current_b.write(ref_fname%c)
        print('[+] Saved reference image: {}'.format(ref_fname%c))
    return [Bitmap(ref_fname%c) for c in range(num_cameras)]

def save_optimized_texture(output_dir, prefix, it_i, data, texture_res):
    texture_fname = join(output_dir, prefix + '_opt_texture_{:04d}.exr'.format(it_i))
    im_data = data.numpy().reshape((texture_res[0], texture_res[1], -1)).astype(np.float32)
    texture_b = Bitmap(im_data[:,:,0], pixel_format=Bitmap.EY)
    texture_b.write(texture_fname)
    return texture_fname

def optimize_rb(config_name, callback=None, use_approx_l1=False, use_interleaving=False,
                output_dir=None, save_intermediate=True):
    """Optimization of RGB colors or textures in the cbox scene."""
    if isinstance(config_name, str):
        config = CONFIGS[config_name]
    else:
        config = config_name
        config_name = config['name']
    num_cameras = config['num_cameras']
    print("Num cameras:", num_cameras)
    output_dir = output_dir or join(OUTPUT_DIR, config_name)
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    forward_spp = config.get('forward_spp', 32)
    backward_spp = config.get('backward_spp', 32)
    max_iterations = config.get('max_iterations', 150)

    def prepare_integrator(allow_approx=True):
        scene_params = config.get('scene_params', [])
        scene = load_scene(config['scene'], parameters=scene_params)
        params = get_differentiable_parameters(scene)
        
        #import code
        #code.interact(local=locals())
        
        params.keep(config['params'])

        # Optimization setup
        integrator_type = integrator_type_for_scene(scene, with_gradients=True, config=config)
        renderer = OptiXRenderer(scene, integrator_type, params=params)
        if allow_approx:
            renderer.set_use_approx_li_1(True)
        return scene, integrator_type, params, renderer, None


    # Render reference image, if missing
    references_b = load_or_render_ref(config)
    for c in range(num_cameras):
        references_b[c].write(join(output_dir, 'ref%03d.exr'%c))
    loss_function = l2_loss
    def compute_loss(current_b, camera=0, cache=None):
        return compute_image_gradients(references_b[camera], current_b, loss_function,
                                       cache=None)

    if 'use_approx_li_1' in config and config['use_approx_l1'] != use_approx_l1:
        raise ValueError('Error: config specifies `use_approx_l1` = {}, but argument = {}'.format(
            config['use_approx_l1'], use_approx_l1))


    scene, integrator_type, params, renderer, _ = prepare_integrator(allow_approx=use_approx_l1)
    
    volume_data = params[DATA_KEY]
    
    learning_rate_factor = 1
    if use_approx_l1:
        print('[i] Using approximate Li = 1')

        # For methods with approximations, estimate the learning rate
        # factor to be applied from the first iteration.
        if 'approx_lr_ratio' in config:
            learning_rate_factor = config['approx_lr_ratio']
        else:
            from approx_utils import estimate_learning_rate_scaling
            learning_rate_factor, _, _ = estimate_learning_rate_scaling(
                config, prepare_integrator, compute_loss)
        print('[i] Using learning rate factor = {}'.format(learning_rate_factor))

    learning_rate = learning_rate_factor * config.get('learning_rate', 2e-2)
    opt = Adam(params, lr=learning_rate)

    if use_interleaving:
        print('[i] Using interleaving (intentional off-by-one)')

    cache = None
    metadata = { k: [] for k in ['timing', 'timings_forward', 'timings_backward', 'loss'] }
    total_time = 0

    if use_interleaving:
        # Render current state for initial image gradients
        current_b = renderer.render(0, spp=forward_spp)

    for it_i in range(max_iterations+1):
        if it_i == 40:
            forward_spp *= 2
            backward_spp *= 2
        camera = np.random.randint(0, num_cameras) #it_i % num_cameras
        scene.set_current_sensor(camera)

        t0 = time.time()

        # Render current state (forward pass)
        seed_forward = it_i
        if not use_interleaving:
            current_b = renderer.render(seed_forward, spp=forward_spp)
        t_end_forward = time.time()

        # Compute image gradients
        image_gradients, loss, cache = compute_image_gradients(
            references_b[camera], current_b, loss_function, cache=None)

        seed_backward = max_iterations + seed_forward + 1
        img_or_none = renderer.delta_render(seed_backward, image_gradients, spp=backward_spp,
                                            return_image=use_interleaving)
        if use_interleaving:
            current_b = img_or_none

        # Apply gradient step
        opt.step()

        ensure_valid_values(params)

        # Re-upload globals to the correct OptiX constants
        renderer.upload_bsdf_globals()

        t_end_backward = time.time()
        elapsed = t_end_backward - t0
        loss_v = loss.numpy()[0]
        metadata['loss'].append(loss_v)
        metadata['timing'].append(elapsed)
        metadata['timings_forward'].append(t_end_forward - t0)
        metadata['timings_backward'].append(t_end_backward - t_end_forward)
        total_time += t_end_backward - t0

        if save_intermediate and ((it_i%50)==0 or it_i<50):
            fname = join(output_dir, 'out_{:04d}.exr'.format(it_i))
            current_b.write(fname)
            data = get_volume_from_params(params)
            save_volume_slices(data, join(output_dir, "slice-%04d-%%s.exr"%it_i)    )
            
            # save volume
            np.savez(join(output_dir, "volume.npz"),
                volume=data,
                time_sec = total_time)

        if callback is not None:
            images_np = [np.array(current_b, copy=True)]
            callback(it_i, params, config['params'], opt, max_iterations,
                     loss, images_np, None)


        print("[{:03d}]: loss = {:.6f}, took {:.2f}s".format(
              it_i, loss_v, elapsed), end='\r')

    del renderer, opt

    # Save metadata about the run
    fname = join(output_dir, 'metadata.csv')
    save_info_csv(fname, **metadata)
    print('[+] Saved run metadata to: {}'.format(fname))

    return metadata['loss'][-1], current_b, metadata

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='optimize_rb.py')
    parser.add_argument('config_name', type=str, help='Configuration (scene) to optimize')
    args = parser.parse_args()

    optimize_rb(args.config_name)
