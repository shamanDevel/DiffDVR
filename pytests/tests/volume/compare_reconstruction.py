"""
Absorption-only Reconstruction,
comparison with other methods (ASTRA + Mitsuba)

Command lines:
Skull: python3 compare_reconstruction.py results/volume/density/skull7absorption config-files/skull7absorption.json --views 64 --diffdvrL1 --visCropSlice 73:2:64:96 --visCropRendering 62:250:192:128 --visRenderingDiffScaling 20 --visSliceDiffScaling 5 --visSliceRotate 3
Plume: python3 compare_reconstruction.py results/volume/density/plume123absorption config-files/plume123-linear-absorption.json --views 64 --diffdvrL1 --visCropSlice 95:125:96:64 --visCropRendering 90:30:192:128 --visRenderingDiffScaling 20 --visSliceDiffScaling 5 --visSliceRotate 2
Thorax: python3 compare_reconstruction.py results/volume/density/thorax2absorption config-files/thorax2absorption.json --views 64 --diffdvrL1 --visSliceIndividualNormalize --visCropSlice 104:37:96:64 --visCropRendering 30:215:192:128 --visRenderingDiffScaling 20 --visSliceDiffScaling 5 --visSliceRotate 0

First run with -amd for the reconstruction,
then run Mitsuba separately,
then visualize the result with -v .

"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import argparse
import imageio
from typing import Optional
import subprocess
import time
import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
from collections import defaultdict

from diffdvr import Renderer, CameraOnASphere, Settings, setup_default_settings, \
    fibonacci_sphere, renderer_dtype_torch, renderer_dtype_np, VolumeDensities, \
    TfTexture, SmoothnessPrior, toCHW, VolumePreshaded, cvector_to_numpy
from losses.lossbuilder import LossBuilder
import pyrenderer

def _lerp(x, a, b):
    return a + x * (b-a)

def _init_parser(parser: argparse.ArgumentParser):
    parser.add_argument('outputFolder', type=str, help="Output folder with results and intermediate data")
    # dataset
    parser.add_argument('settingsFile', type=str, help="Settings .json file")

    parser.add_argument('-a', action='store_true', help="Run ASTRA reconstruction")
    parser.add_argument('-m', action='store_true', help="Run/prepare Mitsuba reconstruction")
    parser.add_argument('-d', action='store_true', help="Run DiffDvr reconstruction")
    parser.add_argument('-v', action='store_true', help="Visualize results")
    parser.add_argument('-r', action='store_true', help="Render and save all reference images")

    parser.add_argument('--views', default=8, type=int, help="Number of views")
    parser.add_argument('--imgres', default=512, type=int, help="Image resolution")
    parser.add_argument('--astraIterations', default=500, type=int)
    parser.add_argument('--diffdvrIterations', default=10, type=int)
    parser.add_argument('--diffdvrLastIterations', default=50, type=int)
    parser.add_argument('--diffdvrMultiscale', default=32, type=int, help="start resolution for multiscale")
    parser.add_argument('--diffdvrL1', action='store_true')
    parser.add_argument('--diffdvrL2', action='store_true')
    parser.add_argument('--diffdvrPriorSmoothing', default=0.5, type=float)
    parser.add_argument('--diffdvrOptimizer', default='Adam', type=str)
    parser.add_argument('--diffdvrLR', default=0.3, type=float)
    parser.add_argument('--diffdvrBatches', default=8, type=int)
    parser.add_argument('--seed', default=124, type=int)

    # for visualization (has no impact on the reconstruction / statistics)
    parser.add_argument('--visSliceIndividualNormalize', action='store_true',
                        help="If specified, each slice is normalized individually by max absorption instead of the whole volume")
    parser.add_argument('--visCropSlice', default=None, type=str,
                        help="Format 'x:y:w:h', specifies crops for the slice images")
    parser.add_argument('--visCropRendering', default=None, type=str,
                        help="Format 'x:y:w:h', specifies crops for the rendered images")
    parser.add_argument('--visCropSliceThickness', default=2, type=int)
    parser.add_argument('--visCropRenderingThickness', default=4, type=int)
    parser.add_argument('--visSliceDiffScaling', default=None, type=float,
                        help="Scaling for the slice difference images. Default=None: normalize to the range")
    parser.add_argument('--visRenderingDiffScaling', default=None, type=float,
                        help="Scaling for the rendering difference images. Default=None: normalize to the range")
    parser.add_argument('--visSliceRotate', default=0, type=int,
                        help="Number of times the slice image is rotated by 90°")

def _prepare_volume(settings_file: str, views: int):
    # settings
    s = Settings(settings_file)
    reference_volume = s.load_dataset()
    reference_volume_data = reference_volume.getDataCpu(0)
    device = reference_volume_data.device
    world_size = cvector_to_numpy(reference_volume.world_size)
    print("world size:", world_size)

    # camera
    camera_config = s.get_camera()
    camera_pitch_cpu, camera_yaw_cpu = fibonacci_sphere(views, dtype=renderer_dtype_np)
    camera_distance_cpu = camera_config.distance * np.ones((views,), dtype=renderer_dtype_np)
    camera_center_cpu = np.stack([camera_config.center] * views, axis=0).astype(dtype=renderer_dtype_np)
    camera_fov_radians = camera_config.fov_y_radians
    camera_module = CameraOnASphere(camera_config.orientation)
    cameras = camera_module(
        torch.from_numpy(camera_center_cpu).to(device=device),
        torch.from_numpy(camera_yaw_cpu).to(device=device).unsqueeze(1),
        torch.from_numpy(camera_pitch_cpu).to(device=device).unsqueeze(1),
        torch.from_numpy(camera_distance_cpu).to(device=device).unsqueeze(1))

    # reference camera
    reference_cameras = camera_module(
        torch.from_numpy(camera_center_cpu[:1]).to(device=device),
        camera_config.yaw_radians * torch.ones((1, 1), dtype=renderer_dtype_torch).to(device=device),
        camera_config.pitch_radians * torch.ones((1, 1), dtype=renderer_dtype_torch).to(device=device),
        camera_config.distance * torch.ones((1, 1), dtype=renderer_dtype_torch).to(device=device))

    # TF
    min_density = s._data["tfEditor"]["minDensity"]
    max_density = s._data["tfEditor"]["maxDensity"]
    opacity_scaling = s._data["tfEditor"]["opacityScaling"]
    g = s._data["tfEditor"]['editorLinear']
    densityAxisOpacity = g['densityAxisOpacity']
    assert len(densityAxisOpacity) == 2
    opacityAxis = g['opacityAxis']
    assert len(opacityAxis) == 2
    actual_min_density = _lerp(densityAxisOpacity[0], min_density, max_density)
    actual_max_density = _lerp(densityAxisOpacity[1], min_density, max_density)
    absorption_scaling = 1  # opacityAxis[1]

    # transform volume data from [actual_min_density, actual_max_density] to [0,1] (with clamping)
    # and then multiply with absorption_scaling
    print("Transform volume data from [%.4f, %.4f] to [0,1] and scale by %.4f" % (
        actual_min_density, actual_max_density, absorption_scaling))
    reference_volume_data = torch.clip(
        (reference_volume_data - actual_min_density) / (actual_max_density - actual_min_density),
        0.0, 1.0, out=reference_volume_data)
    reference_volume_data *= absorption_scaling

    return {
        'settings': s,
        'reference_volume': reference_volume,
        'reference_volume_data': reference_volume_data,
        'cameras': cameras,
        'camera_fov_radians': camera_fov_radians,
        'world_size': world_size,
        'reference_cameras': reference_cameras,
    }

def _setup_renderer(data:dict, resolution: int, with_reference_camera=False):
    cuda_device = torch.device("cuda")
    rs = setup_default_settings(
        data['reference_volume'], resolution, resolution,
        data['settings'].get_stepsize(), False)
    rs.tf_mode = pyrenderer.TFMode.Identity
    tf_reference = torch.tensor([[
        # r,g,b,a,pos
        [1, 1]
    ]], dtype=renderer_dtype_torch, device=cuda_device)
    if with_reference_camera:
        cameras = torch.cat([data['reference_cameras'], data['cameras']])
    else:
        cameras = data['cameras']
    cameras_cuda = cameras.to(device=cuda_device)
    renderer = Renderer(rs, optimize_volume=True,
                        gradient_method='adjoint')

    return rs, cameras_cuda, tf_reference, renderer

def _render_reference(data:dict, output_path_template: Optional[str], resolution: int):
    cuda_device = torch.device("cuda")
    volume_data_cuda = data['reference_volume_data'].to(cuda_device)
    rs, cameras_cuda, tf_reference, renderer = \
        _setup_renderer(data, resolution)

    reference_images = renderer(
        camera=cameras_cuda, fov_y_radians=data['camera_fov_radians'],
        tf=tf_reference, volume=volume_data_cuda)
    reference_images = toCHW(reference_images).detach()

    if output_path_template is not None:
        absorption_images = torch.stack([reference_images[:, 3, :, :]] * 3, dim=-1).cpu().numpy()
        max_absorption = np.max(absorption_images)
        for v in range(cameras_cuda.shape[0]):
            absorption_image = (absorption_images[v] / max_absorption * 255).astype(np.uint8)
            imageio.imwrite(output_path_template % v, absorption_image)

    return reference_images

def _call_astra(cfg: dict, folder: str, resolution: int, iterations: int):
    # export to numpy for ASTRA
    print("export to numpy for ASTRA")
    astra_input_file = os.path.join(folder, "astra-input.npz")
    astra_output_file = os.path.join(folder, "astra-output.npz")
    np.savez(astra_input_file,
             volume=cfg['reference_volume_data'],
             cameras=cfg['cameras'],
             fov_y_radians=cfg['camera_fov_radians'],
             world_size=cfg['world_size'])

    # call astra
    cwd = os.path.abspath(os.path.join(os.path.split(__file__)[0], "astra"))
    print("working directory:", cwd)
    args = [
        "conda", "run", "-n", "py37astra",
        "--cwd", cwd,
        "python", "VolumeReconstruction.py",
        "--output", astra_output_file,
        "--input", astra_input_file,
        "--resolution", "%d"%resolution,
        "--iterations", "%d"%iterations,
    ]
    ret = subprocess.run(args)

def _call_mitsuba(cfg: dict, folder: str, resolution: int, opt):
    # export volumes
    mitsuba_scene = os.path.join(folder, "mitsuba_scene.xml")
    mitsuba_cfg = os.path.join(folder, "mitsuba_cfg.py")
    mitsuba_reference = os.path.join(folder, "mitsuba_reference-%03d.exr")
    mitsuba_volume_reference = os.path.join(folder, "mitsuba_reference.vol")
    mitsuba_volume_initial = os.path.join(folder, "mitsuba_initial.vol")

    volume = cfg['reference_volume_data'].cpu().numpy()[0]
    print("Strides:", volume.strides)
    cameras = cfg['cameras'].cpu().numpy()
    fov_y_radians = cfg['camera_fov_radians']
    world_size = cfg['world_size']
    num_cameras = cameras.shape[0]

    def write_grid_binary_data(filename, values):
        values = np.array(values)
        with open(filename, 'wb') as f:
            f.write(b'V')
            f.write(b'O')
            f.write(b'L')
            f.write(np.uint8(3).tobytes())  # Version
            f.write(np.int32(1).tobytes())  # type
            f.write(np.int32(volume.shape[0]).tobytes())  # size
            f.write(np.int32(volume.shape[1]).tobytes())
            f.write(np.int32(volume.shape[2]).tobytes())
            f.write(np.int32(1).tobytes())  # channels
            f.write(np.float32(0.0).tobytes())  # bbox
            f.write(np.float32(0.0).tobytes())
            f.write(np.float32(0.0).tobytes())
            f.write(np.float32(1.0).tobytes())
            f.write(np.float32(1.0).tobytes())
            f.write(np.float32(1.0).tobytes())
            f.write(values.astype(np.float32).tobytes())

    write_grid_binary_data(mitsuba_volume_reference, volume)
    write_grid_binary_data(mitsuba_volume_initial, 0.1 * np.ones_like(volume))

    # write config
    with open(os.path.join(os.path.split(__file__)[0], "mitsuba/config.py.template"), "r") as f:
        config_template = f.read()
    config_template = config_template.replace("{$SCENE$}", mitsuba_scene)
    config_template = config_template.replace("{$REF_NAME$}", mitsuba_reference)
    config_template = config_template.replace("{$NUM_CAMERAS$}", str(num_cameras))
    config_template = config_template.replace("{$IMG_RES$}", str(resolution))
    config_template = config_template.replace("{$INITIAL_VOLUME$}", mitsuba_volume_initial)
    config_template = config_template.replace("{$REFERENCE_VOLUME$}", mitsuba_volume_reference)
    with open(mitsuba_cfg, 'w') as f:
        f.write(config_template)

    # write scene
    with open(os.path.join(os.path.split(__file__)[0], "mitsuba/scene.xml.template"), "r") as f:
        scene_template = f.read()
    # TODO: world size in <transform> of the medium
    camera_strings = []
    camera_template = """
    <sensor type="perspective" id="{id}">
        <string name="fov_axis" value="smaller"/>
        <float name="near_clip" value="0.1"/>
        <float name="far_clip" value="2800"/>
        <float name="focus_distance" value="1000"/>
        <float name="fov" value="{fov:.5f}"/>
        <transform name="to_world">
            <lookat origin="{originx:.5f}, {originy:.5f}, {originz:.5f}"
                    target="{targetx:.5f}, {targety:.5f}, {targetz:.5f}"
                    up    ="{upx:.5f}, {upy:.5f}, {upz:.5f}"/>
        </transform>

        <ref id="sampler"/>
        <ref id="film"/>
    </sensor>
    """
    for c in range(num_cameras):
        src = cameras[c, 0, :]
        right = cameras[c, 1, :]
        up = cameras[c, 2, :]
        front = np.cross(up, right)
        lookat = src+front
        camera_strings.append(camera_template.format(
            id=c,
            fov=np.rad2deg(fov_y_radians),
            originx=src[0],
            originy=src[1],
            originz=src[2],
            targetx=lookat[0],
            targety=lookat[1],
            targetz=lookat[2],
            upx=up[0],
            upy=up[1],
            upz=up[2],
        ))
    scene_template = scene_template.replace("{$SENSORS$}", "\n".join(camera_strings))
    with open(mitsuba_scene, 'w') as f:
        f.write(scene_template)

    # THEN: Run optimize_rb2.py in your Mitsuba environment
    # Copy the output volume.npz into the output folder and
    # rename it to "mitsuba-output.npz".

def _run_diffdvr_reconstruction(cfg: dict, folder: str, resolution: int, opt):
    diffdvr_output_file = os.path.join(folder, "diffdvr-output.npz")

    settings = cfg['settings']
    volume=cfg['reference_volume_data']
    cameras=cfg['cameras']
    fov_y_radians=cfg['camera_fov_radians']
    world_size=cfg['world_size']
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.set_num_threads(4)
    views = cameras.shape[0]

    # this is a simplified version of train_volume
    multiscale = opt.diffdvrMultiscale
    if multiscale > 0:
        start_resolution = multiscale
        end_resolution = volume.shape[1]
        mipmap_levels = int(np.round(np.log2(end_resolution/start_resolution)))
        assert start_resolution*(2**mipmap_levels) == end_resolution,  \
            "the end resolution is not a power of the start resolution"
        print("start resolution:", start_resolution, ", end resolution:", end_resolution,
              "->", mipmap_levels, "mipmap levels")
    else:
        start_resolution = end_resolution = volume.shape[1]
        mipmap_levels = 0

    rs, cameras_cuda, tf_reference, renderer = \
        _setup_renderer(cfg, resolution)
    cuda_device = torch.device("cuda")
    volume_data_cuda = cfg['reference_volume_data'].to(cuda_device)
    reference_images = renderer(
        camera=cameras_cuda, fov_y_radians=cfg['camera_fov_radians'],
        tf=tf_reference, volume=volume_data_cuda)
    reference_images = reference_images.detach()[:,:,:,3]

    # volume
    volume_channels = 1
    volume_densities = VolumeDensities()
    #initial_volume_parameterized = torch.randn(
    #    (1, start_resolution, start_resolution, start_resolution),
    #    dtype=renderer_dtype_torch, device=cuda_device)
    initial_volume_parameterized = VolumeDensities.prepare_input(0.1 * torch.ones(
        (1, start_resolution, start_resolution, start_resolution),
        dtype=renderer_dtype_torch, device=cuda_device))

    class LossNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if opt.diffdvrL1 == opt.diffdvrL2:
                raise ValueError("Either L1 or L2, not both")
            if opt.diffdvrL1:
                self._loss = torch.nn.L1Loss()
            else:
                self._loss = torch.nn.MSELoss()
        def forward(self, x, y):
            return self._loss(x, y)

    loss = LossNet()
    loss.to(device=cuda_device)

    class PriorLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._ps = SmoothnessPrior((1,2,3))

        def forward(self, volume):
            return self._ps(volume)

    prior_loss = PriorLoss()
    prior_loss.to(device=cuda_device, dtype=renderer_dtype_torch)
    prior_loss_weight = opt.diffdvrPriorSmoothing

    # optimization
    epochs = opt.diffdvrIterations
    lastEpochs = opt.diffdvrLastIterations
    batch_size = opt.diffdvrBatches
    num_batches = int(np.ceil(views / batch_size))
    optimizer_class = getattr(torch.optim, opt.diffdvrOptimizer)

    # dataloader
    dataset_train = dict([
        (j, (reference_images[j], cameras[j])) for j in range(views)
    ])
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)

    last_volume = None
    last_loss = 0
    print("Optimize")
    tstart = time.time()
    # loop over mipmap-levels
    current_volume_parameterized = None
    for mipmap in range(0, mipmap_levels + 1):
        current_resolution = int(start_resolution * (2 ** mipmap))

        if mipmap == 0:
            current_volume_parameterized = initial_volume_parameterized.clone()
        else:
            current_volume_parameterized = F.interpolate(
                current_volume_parameterized.detach().unsqueeze(0),
                (current_resolution, current_resolution, current_resolution),
                mode='trilinear', align_corners=True)[0].clone()
        current_volume_parameterized.requires_grad_(True)
        optimizer = optimizer_class([current_volume_parameterized], lr=opt.diffdvrLR)

        view_indices = np.arange(views)
        tf_indices = np.arange(views)
        if mipmap == mipmap_levels:
            epochs = lastEpochs
        pbar = tqdm.tqdm(range(epochs))
        for iteration in pbar:
            pbar.set_description("Mipmap %d, loss %.8f"%(mipmap, last_loss))
            last_loss = 0
            for current_references,current_cameras in dataloader_train:
                current_references = current_references.to(device=cuda_device)
                current_cameras = current_cameras.to(device=cuda_device)

                def optim_closure():
                    nonlocal last_volume, last_loss
                    optimizer.zero_grad()
                    volume = volume_densities(current_volume_parameterized)
                    images = renderer(
                        camera=current_cameras, fov_y_radians=cfg['camera_fov_radians'],
                        tf=tf_reference, volume=volume)[:,:,:,3]
                    if prior_loss_weight > 0:
                        loss_value = loss(images, current_references) + prior_loss_weight * prior_loss(volume)
                    else:
                        loss_value = loss(images, current_references)
                    loss_value.backward()
                    last_loss += loss_value.item()
                    last_volume = volume.detach()
                    return loss_value

                optimizer.step(optim_closure)

    tend = time.time()
    print("Done in",(tend-tstart),"seconds, loss:", last_loss)

    output_images = renderer(
        camera=cameras_cuda, fov_y_radians=cfg['camera_fov_radians'],
        tf=tf_reference, volume=last_volume)
    output_images = output_images.detach()[:, :, :, 3]
    np.savez(diffdvr_output_file,
             volume=last_volume[0].detach().cpu().numpy(),
             input_images=reference_images.cpu().numpy(),
             output_images=output_images.cpu().numpy(),
             time_sec=(tend - tstart))

def _analyze(cfg: dict, opt, folder: str, save:bool):
    KEYS = ["astra", "diffdvr", "mitsuba"]
    npzfiles = [np.load(os.path.join(folder,"%s-output.npz"%key)) for key in KEYS]
    X = np.arange(len(KEYS))
    original_volume = cfg['reference_volume_data'][0]
    original_volume_np = original_volume.numpy()

    def autolabel(rects, ax, format='%d', log=False):
        """
        Attach a text label above each bar displaying its height
        """
        max_height = max([rect.get_height() for rect in rects])
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.,
                    (1.05 * height) if log else (height + 0.01 * max_height),
                    format % height,
                    ha='center', va='bottom')

    # timings + L2-loss on the volume
    fig, ax = plt.subplots(1, 2, sharex=True)
    Y = [f['time_sec'] for f in npzfiles]
    rects = ax[0].bar(X, Y, tick_label=KEYS, log=True)
    autolabel(rects, ax[0], log=True)
    ax[0].set_title("Timings (sec, log-scale)")
    Y = [F.mse_loss(torch.from_numpy(f['volume']), original_volume) for f in npzfiles]
    rects = ax[1].bar(X, Y, tick_label=KEYS, log=False)
    autolabel(rects, ax[1], '%.5f')
    ax[1].set_title("L2-Loss on the Volume")
    fig.subplots_adjust(
            top=0.925,
            bottom=0.07,
            left=0.09,
            right=0.955,
            hspace=0.2,
            wspace=0.345)
    if save:
        plt.savefig(os.path.join(folder, "results-timing+error.png"))
        plt.close(fig)

    # Renderings
    rs, cameras_cuda, tf_reference, renderer = \
        _setup_renderer(cfg, 512, with_reference_camera=True)
    cuda_device = torch.device("cuda")
    input_images = renderer(
        camera=cameras_cuda, fov_y_radians=cfg['camera_fov_radians'],
        tf=tf_reference, volume=original_volume.to(device=cuda_device).unsqueeze(0))
    input_images = input_images.detach()[:, :, :, 3].cpu().numpy()
    max_absorption = np.max(input_images)
    input_images /= max_absorption
    times = 4

    class SSIM():
        def __init__(self):
            self._ssim = LossBuilder(torch.device("cpu")).ssim_loss(1)
        def __call__(self, x, y):
            return self._ssim(torch.from_numpy(x).unsqueeze(0).unsqueeze(0), torch.from_numpy(y).unsqueeze(0).unsqueeze(0)).item()
    ssim = SSIM()
    class PSNR():
        def __call__(self, x, y):
            return 10 * torch.log10(1 / torch.nn.functional.mse_loss(
                torch.from_numpy(x), torch.from_numpy(y), reduction='mean')).item()
    psnr = PSNR()

    # render images
    output_images = []
    image_psnrx = []
    image_ssimx = []
    max_difference = 0
    for x in X:
        output_image = renderer(
            camera=cameras_cuda, fov_y_radians=cfg['camera_fov_radians'],
            tf=tf_reference, volume=torch.from_numpy(npzfiles[x]['volume']).to(device=cuda_device).unsqueeze(0))
        output_image = output_image.detach()[:, :, :, 3].cpu().numpy()
        output_image /= max_absorption
        output_images.append(output_image)
        max_difference = max(max_difference, np.max(np.abs(input_images-output_image)))

    print("max rendering difference:", max_difference)
    if opt.visRenderingDiffScaling is not None:
        print("overwrite to a scaling of", opt.visRenderingDiffScaling)
        max_difference = 1 / opt.visRenderingDiffScaling

    def cropRendering(img, output, mark=False):
        if opt.visCropRendering is None: return
        img = np.copy(img)[:,:,:3]
        thickness = opt.visCropRenderingThickness
        off = thickness//2
        thickness = thickness - off
        x,y,w,h = tuple(map(int, opt.visCropRendering.split(':')))
        color = np.array([100, 0, 0], dtype=np.uint8)
        # draw markings into crop
        img[y - off:y + h + thickness, x - off:x + thickness, :] = color
        img[y - off:y + h + thickness, x + w - off:x + w + thickness, :] = color
        img[y - off:y + thickness, x - off:x + w + thickness, :] = color
        img[y + h - off:y + h + thickness, x - off:x + w + thickness, :] = color
        if mark:
            # save directly
            imageio.imwrite(output, img)
        else:
            # crop
            img = img[y-off: y+h+thickness, x-off : x+w+thickness, :]
            imageio.imwrite(output, img)

    # save input images
    if save:
        for t in range(times):
            img = (np.clip(1-np.stack([input_images[t]]*3, axis=2), 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(
                os.path.join(folder, "results-rendering-reference-%d.png" % t), img)
            cropRendering(img, os.path.join(folder, "results-rendering-reference-%d-crop.png" % t))
            cropRendering(img, os.path.join(folder, "results-rendering-reference-%d-mark.png" % t), mark=True)

    cmap = matplotlib.cm.get_cmap('bwr')
    norm = matplotlib.colors.Normalize(-1, +1, clip=True)
    for x in X:
        fig, ax = plt.subplots(3, times, sharex=True, sharey=True)
        ax[0][0].set_ylabel("Reference")
        ax[1][0].set_ylabel("Reconstruction")
        ax[2][0].set_ylabel("Difference")
        fig.suptitle("Rendering %s, max difference: %e" % (KEYS[x], max_difference))
        image_ssimx.append([])
        image_psnrx.append([])
        for t in range(times):
            diff_img = (output_images[x][t]-input_images[t])/max_difference
            ax[0][t].set_title("View %d"%t)
            input3 = np.stack([input_images[t]]*3, axis=2)
            output3 = np.stack([output_images[x][t]] * 3, axis=2)
            ax[0][t].imshow(input3)
            ax[1][t].imshow(output3)
            ax[2][t].imshow(diff_img, cmap=cmap, vmin=-1, vmax=+1)
            image_ssimx[-1].append(ssim(input_images[t], output_images[x][t]))
            image_psnrx[-1].append(psnr(input_images[t], output_images[x][t]))
            if save:
                img = (np.clip(1 - output3, 0, 1) * 255).astype(np.uint8)
                imageio.imwrite(
                    os.path.join(folder, "results-rendering-%s-%d.png" % (KEYS[x], t)), img)
                cropRendering(img, os.path.join(folder, "results-rendering-%s-%d-crop.png" % (KEYS[x], t)))
                img = (cmap(norm(diff_img)) * 255).astype(np.uint8)
                imageio.imwrite(
                    os.path.join(folder, "results-rendering-%s-%d-diff.png" % (KEYS[x], t)), img)
                cropRendering(img, os.path.join(folder, "results-rendering-%s-%d-diff-crop.png" % (KEYS[x], t)))
        if save:
            plt.savefig(os.path.join(folder, "results-fig-rendering-%s.png"%KEYS[x]))
            plt.close(fig)

    # Slices

    def cropSlices(img, output, mark=False):
        if opt.visCropSlice is None: return
        img = np.copy(img)[:,:,:3]
        thickness = opt.visCropSliceThickness
        off = thickness//2
        thickness = thickness - off
        x,y,w,h = tuple(map(int, opt.visCropSlice.split(':')))
        color = np.array([0, 100, 0], dtype=np.uint8)
        # draw markings into crop
        img[y - off:y + h + thickness, x - off:x + thickness, :] = color
        img[y - off:y + h + thickness, x + w - off:x + w + thickness, :] = color
        img[y - off:y + thickness, x - off:x + w + thickness, :] = color
        img[y + h - off:y + h + thickness, x - off:x + w + thickness, :] = color
        if mark:
            # save directly
            if opt.visSliceRotate>0:
                img = np.rot90(img, k=opt.visSliceRotate)
            imageio.imwrite(output, img)
        else:
            # crop
            img = img[y-off: y+h+thickness, x-off : x+w+thickness, :]
            if opt.visSliceRotate>0:
                img = np.rot90(img, k=opt.visSliceRotate)
            imageio.imwrite(output, img)

    sliceIndividualNormalize = opt.visSliceIndividualNormalize
    slices = list(np.linspace(0, original_volume.shape[-1], 7)[2:-2])
    if sliceIndividualNormalize:
        max_slice_difference = defaultdict(lambda: 0.0)
        max_slice_absorption = defaultdict(lambda: 0.0)
        for x in X:
            vol = npzfiles[x]['volume']
            for t, s in enumerate(slices):
                s = int(s)
                slice = vol[s, ...]
                orig_slice = original_volume_np[s, ...]
                diff = np.max(np.abs(slice - orig_slice))
                max_slice_difference[s] = max(max_slice_difference[s], diff)
                max_slice_absorption[s] = max(max_slice_absorption[s], np.max(slice))
                max_slice_absorption[s] = max(max_slice_absorption[s], np.max(orig_slice))
    else:
        max_slice_difference_0 = 0
        for x in X:
            vol = npzfiles[x]['volume']
            max_slice_difference_0 = max(max_slice_difference_0, np.max(np.abs(vol - original_volume_np)))
        max_slice_difference = defaultdict(lambda: max_slice_difference_0)
        max_slice_absorption = defaultdict(lambda: 1.0)
    print("max slice difference:", [max_slice_difference[int(s)] for s in slices])
    if opt.visSliceDiffScaling is not None:
        print("overwrite to a scaling of", opt.visSliceDiffScaling)
        max_slice_difference = defaultdict(lambda: 1/opt.visSliceDiffScaling)

    volume_psnrx = []
    if save:
        for t,s in enumerate(slices):
            s = int(s)
            orig_slice = original_volume_np[s, ...] / max_slice_absorption[s]
            img = (np.clip(1-np.stack([orig_slice]*3, axis=2), 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(
                os.path.join(folder, "results-slice-reference-%d.png" % t), img)
            cropSlices(img, os.path.join(folder, "results-slice-reference-%d-crop.png" % t))
            cropSlices(img, os.path.join(folder, "results-slice-reference-%d-mark.png" % t), mark=True)

    for x in X:
        fig, ax = plt.subplots(3, len(slices), sharex=True, sharey=True)
        ax[0][0].set_ylabel("Reference")
        ax[1][0].set_ylabel("Reconstruction")
        ax[2][0].set_ylabel("Difference")
        vol = npzfiles[x]['volume']
        volume_psnrx.append(psnr(vol, original_volume_np))
        fig.suptitle("Slices %s, max difference: %e" % (KEYS[x], max_slice_difference[int(slices[0])]))
        for t,s in enumerate(slices):
            s = int(s)
            slice = vol[s,...]
            orig_slice = original_volume_np[s,...]
            diff = (slice - orig_slice) / max_slice_difference[s]
            ax[0][t].set_title("Slice %d" % s)
            ax[0][t].imshow(np.stack([orig_slice / max_slice_absorption[s]] * 3, axis=2))
            ax[1][t].imshow(np.stack([slice / max_slice_absorption[s]] * 3, axis=2))
            ax[2][t].imshow(diff, cmap=cmap, vmin=-1, vmax=+1)
            if save:
                img = (np.clip(1 - np.stack([slice / max_slice_absorption[s]] * 3, axis=2), 0, 1) * 255).astype(np.uint8)
                imageio.imwrite(
                    os.path.join(folder, "results-slice-%s-%d.png" % (KEYS[x], t)), img)
                cropSlices(img, os.path.join(folder, "results-slice-%s-%d-crop.png" % (KEYS[x], t)))
                img = (cmap(norm(diff)) * 255).astype(np.uint8)
                imageio.imwrite(
                    os.path.join(folder, "results-slice-%s-%d-diff.png" % (KEYS[x], t)), img)
                cropSlices(img, os.path.join(folder, "results-slice-%s-%d-diff-crop.png" % (KEYS[x], t)))
        if save:
            plt.savefig(os.path.join(folder, "results-fig-slice-%s.png"%KEYS[x]))
            plt.close(fig)

    if save:
        plt.close("all")
    else:
        plt.show()

    # PSNR / SSIM (for latex)
    LATEX_IMAGE_INDEX = 0
    INDEX_DIFFDVR = KEYS.index("diffdvr")
    INDEX_ASTRA = KEYS.index("astra")
    INDEX_MITSUBA = KEYS.index("mitsuba")
    volume_psnrx = np.array(volume_psnrx)
    image_psnrx = np.array(image_psnrx)
    image_ssimx = np.array(image_ssimx)
    with open(os.path.join(folder, "results-PsnrSsim.tex"), "w") as f:
        def strBold(array, key, format):
            s = format % (array[key])
            if np.argmax(array) == key:
                return "\\textbf{%s}"%s
            return s
        f.write("& & PSNR: %s & PSNR: %s & PSNR: %s" % (
            strBold(volume_psnrx, INDEX_DIFFDVR, "%.2fdB"), strBold(volume_psnrx, INDEX_ASTRA, "%.2fdB"), strBold(volume_psnrx, INDEX_MITSUBA, "%.2fdB")
        ))
        f.write("& & & PSNR: %s & PSNR: %s & PSNR: %s" % (
            strBold(image_psnrx[:,LATEX_IMAGE_INDEX], INDEX_DIFFDVR, "%.2fdB"), strBold(image_psnrx[:,LATEX_IMAGE_INDEX], INDEX_ASTRA, "%.2fdB"), strBold(image_psnrx[:,LATEX_IMAGE_INDEX], INDEX_MITSUBA, "%.2fdB")
        ))
        f.write("\\\\\n&&&&")
        f.write("& & & SSIM: %s & SSIM: %s & SSIM: %s" % (
            strBold(image_ssimx[:,LATEX_IMAGE_INDEX], INDEX_DIFFDVR, "%.4f"), strBold(image_ssimx[:,LATEX_IMAGE_INDEX], INDEX_ASTRA, "%.4f"), strBold(image_ssimx[:,LATEX_IMAGE_INDEX], INDEX_MITSUBA, "%.4f")
        ))
        f.write("\n")

def _main():
    parser = argparse.ArgumentParser(
        description='Reconstruction comparison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _init_parser(parser)
    opt = parser.parse_args()
    folder = os.path.abspath(opt.outputFolder)
    os.makedirs(folder, exist_ok=True)
    print("output folder:", folder)

    cfg = _prepare_volume(opt.settingsFile, opt.views)

    if opt.r:
        print("Render reference")
        _render_reference(cfg, os.path.join(folder, "reference-camera%03d.png"), opt.imgres)

    if opt.a:
        print("Call ASTRA")
        _call_astra(cfg, folder, opt.imgres, opt.astraIterations)

    if opt.m:
        print("Call Mitsuba")
        _call_mitsuba(cfg, folder, opt.imgres, opt)

    if opt.d:
        print("Run DiffDVR reconstruction")
        _run_diffdvr_reconstruction(cfg, folder, opt.imgres, opt)

    if opt.v:
        print("Analyze / Visualize")
        _analyze(cfg, opt, folder, save=True)

if __name__ == '__main__':
    _main()