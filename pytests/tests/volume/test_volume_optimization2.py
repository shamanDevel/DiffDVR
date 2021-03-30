import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from diffdvr import Renderer, VolumeDensities, TfPiecewiseLinear, CameraOnASphere, \
  SmoothnessPrior, toCHW

import pyrenderer

def optimize():
  np.random.seed(42)
  torch.random.manual_seed(42)
  pyrenderer.set_cuda_sync_mode(False)
  pyrenderer.set_cuda_debug_mode(False)

  resolution = 256 # 64
  print("Create Marschner Lobb")
  reference_volume = pyrenderer.Volume.create_implicit(
    pyrenderer.ImplicitEquation.MarschnerLobb, resolution)
  reference_volume.copy_to_gpu()
  print("density tensor: ", reference_volume.getDataGpu(0).shape, reference_volume.getDataGpu(0).dtype, reference_volume.getDataGpu(0).device)

  run_on_cuda = True
  H = 512  # screen height
  W = 512  # screen width
  Y = reference_volume.resolution.y
  Z = reference_volume.resolution.z
  X = reference_volume.resolution.x
  device = reference_volume.getDataGpu(0).device if run_on_cuda else reference_volume.getDataCpu(0).device
  dtype = reference_volume.getDataGpu(0).dtype
  reference_volume_data = reference_volume.getDataGpu(0) if run_on_cuda else reference_volume.getDataCpu(0)

  # optimization / configuration
  optimizer_class = torch.optim.Adam
  iterations = 200
  lr = 0.5
  filename = None #"test_volume_optimization02.mp4"
  video_length = 10  # seconds
  write_to_tensorboard = False # should the frames be written to tensorboard
  image_log_frequency = 10 # every 10 frames
  smoothness_prior_weight = 0.2 # Importance of the smoothness prior, in [0,1]

  # create writer
  if write_to_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter("runs/volume_optimization/"+current_time+"/")
  else:
    writer = None

  # renderer and basic settings
  renderer = Renderer(optimize_volume=True)
  renderer.settings.screen_size = pyrenderer.int2(W, H)
  renderer.settings.volume_filter_mode = pyrenderer.VolumeFilterMode.Trilinear
  renderer.settings.box_min = pyrenderer.real3(-0.5, -0.5, -0.5)
  renderer.settings.box_size = pyrenderer.real3(1, 1, 1)
  renderer.settings.step_size = 0.1 / X
  renderer.settings.tf_mode = pyrenderer.TFMode.Linear
  renderer.settings.blend_mode = pyrenderer.BlendMode.BeerLambert

  # cameras
  num_cameras = 12
  camera_fov_radians = np.radians(45.0)
  camera_center, camera_yaw, camera_pitch, camera_dist = \
    CameraOnASphere.random_points(num_cameras)
  camera_module = CameraOnASphere(pyrenderer.Orientation.Ym)
  cameras = camera_module(camera_center, camera_yaw, camera_pitch, camera_dist)
  cameras = cameras.to(device=device)

  # TFs
  num_tfs = 8
  opacity_scaling = 1.0
  tfs = []
  for _ in range(num_tfs):
    num_peaks = np.random.random_integers(1, 4)
    colors = np.random.uniform(0, 1, (num_peaks, 3))
    widths = np.random.uniform(0.02,0.05, (num_peaks, 1))
    opacities = np.random.uniform(0.5, 1.0, (num_peaks, 1)) / widths
    positions = np.random.uniform(0, 1, (num_peaks, 1))
    peaks = np.concatenate([colors, opacities, positions, widths], axis=1)
    tf = TfPiecewiseLinear.create_from_peaks(peaks, opacity_scaling)
    tfs.append(tf.to(device=device))

  print("inputs created")

  # model, loss functions
  volume_densities = VolumeDensities()
  initial_volume = torch.randn_like(reference_volume_data)
  loss_supervised = torch.nn.MSELoss()
  loss_unsupervised = SmoothnessPrior((1,2,3)) # smooth over X, Y, Z

  # render reference, batch over TFs
  print("Render reference")
  reference_color = [None] * num_tfs
  for i in range(num_tfs):
    reference_color[i] = renderer(
      camera = cameras, fov_y_radians = camera_fov_radians,
      tf = tfs[i], volume = reference_volume_data).detach()

  if writer is not None:
    writer.add_image('gt', torchvision.utils.make_grid(
      toCHW(torch.cat(reference_color, 0)),
      nrow=num_cameras))

  # render initial, batch over TFs
  print("Render initial")
  initial_volume_data = volume_densities(initial_volume)
  initial_color = [None] * num_tfs
  for i in range(num_tfs):
    initial_color[i] = renderer(
      camera=cameras, fov_y_radians=camera_fov_radians,
      tf=tfs[i], volume=initial_volume_data).detach()

  initial_img = torchvision.utils.make_grid(
      toCHW(torch.cat(initial_color, 0)),
      nrow=num_cameras)
  if writer is not None:
    writer.add_image('initial', initial_img)
  #torchvision.utils.save_image(initial_img, "optim-Initial.png")
  del initial_img
  del initial_color
  del initial_volume_data

  # optimize
  print("Optimize")
  current_volume = initial_volume.clone()
  current_volume.requires_grad_()
  optimizer = optimizer_class([current_volume], lr=lr)
  for iteration in range(iterations):
    is_first_closure = True
    def closure():
      nonlocal is_first_closure
      optimizer.zero_grad()
      loss1 = 0
      loss2 = 0
      current_volume_data = volume_densities(current_volume)
      current_color = [None] * num_tfs
      for i in range(num_tfs):
        c = renderer(
          camera=cameras, fov_y_radians=camera_fov_radians,
          tf=tfs[i], volume=current_volume_data)
        loss1 = loss1 + loss_supervised(c, reference_color[i])
        if is_first_closure:
          current_color[i] = c.detach()
      loss2 = loss_unsupervised(current_volume_data)
      loss = (1-smoothness_prior_weight) * loss1 + smoothness_prior_weight * loss2
      loss.backward()

      if is_first_closure:
        is_first_closure = False
        loss_value = loss.item()
        print("Iteration % 04d, total loss: %.7f" % (iteration, loss_value))
        if writer is not None:
          if (iteration % image_log_frequency==0) or (iteration == (iterations-1)):
            img = torchvision.utils.make_grid(
              utils.toCHW(torch.cat(current_color, 0)),
              nrow=num_cameras)
            writer.add_image('recon', img, global_step=iteration)
            print("Image saved")
          writer.add_scalar('loss', loss_value, global_step=iteration)
          writer.add_scalar('l2', loss1.item(), global_step=iteration)
          writer.add_scalar('prior', loss2.item(), global_step=iteration)

      return loss
    optimizer.step(closure=closure)

  if writer is not None:
    writer.close()

  print("Visualize")
  fig, axs = plt.subplots(num_tfs, num_cameras,
                          figsize=(2*num_cameras, 2*num_tfs+0.5),
                          squeeze=False)
  fig.suptitle("Reference")
  for i in range(num_tfs):
    for j in range(num_cameras):
      axs[i, j].imshow(reference_color[i].detach()[j].cpu().numpy()[:,:,:3])
      axs[i, j].set_xticks([])
      axs[i, j].set_yticks([])
  plt.tight_layout()

  #render and visualize reconstruction
  reconstruction_volume_data = volume_densities(current_volume)
  reconstruction_color = [None] * num_tfs
  for i in range(num_tfs):
    reconstruction_color[i] = renderer(
      camera=cameras, fov_y_radians=camera_fov_radians,
      tf=tfs[i], volume=reconstruction_volume_data).detach()
  fig, axs = plt.subplots(num_tfs, num_cameras,
                          figsize=(2 * num_cameras, 2 * num_tfs + 0.5),
                          squeeze=False)
  fig.suptitle("Reconstruction")
  for i in range(num_tfs):
    for j in range(num_cameras):
      axs[i, j].imshow(reconstruction_color[i].detach()[j].cpu().numpy()[:, :, :3])
      axs[i, j].set_xticks([])
      axs[i, j].set_yticks([])
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  optimize()