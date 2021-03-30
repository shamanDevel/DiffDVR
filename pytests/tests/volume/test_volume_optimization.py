import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation
import tqdm

# load pyrenderer
import diffdvr
import pyrenderer

if __name__=='__main__':
  print(pyrenderer.__doc__)

  print("Create Marschner Lobb")
  volume = pyrenderer.Volume.create_implicit(pyrenderer.ImplicitEquation.MarschnerLobb, 64)
  volume.copy_to_gpu()
  print("density tensor: ", volume.getDataGpu(0).shape, volume.getDataGpu(0).dtype, volume.getDataGpu(0).device)

  run_on_cuda = False
  H = 256 # screen height
  W = 256 # screen width
  Y = volume.resolution.y
  Z = volume.resolution.z
  X = volume.resolution.x
  device = volume.getDataGpu(0).device if run_on_cuda else volume.getDataCpu(0).device
  dtype = diffdvr.renderer_dtype_torch

  # optimization
  optimizer_class = torch.optim.Adam
  iterations = 100
  lr = 0.1
  filename = "test_volume_optimization02.mp4"
  video_length = 10 # seconds

  # volume
  def inverseSigmoid(y):
    return torch.log(-y / (y - 1))
  reference_volume = inverseSigmoid(volume.getDataGpu(0) if run_on_cuda else volume.getDataCpu(0))
  #random in [-0.5, +0.5], will be processed by a Sigmoid
  initial_volume = torch.rand(1, X, Y, Z, dtype=dtype, device=device) - 0.5

  # cameras
  camera_orientation = pyrenderer.Orientation.Zp
  camera_fov_radians = np.radians(45.0)
  cameras = [
    # (pitch-degree, yaw-degree, zoom-factor)
    (26, 109, 7),
    (41.5, 46.5, 7),
    (-3, 44, 6),
    (-66, -38, 6),
  ]
  B = len(cameras)
  camera_center = torch.tensor([[0.0, 0.0, 0.0]]*B, dtype=dtype, device=device)
  camera_yaws = torch.tensor([
    [np.radians(pyz[1])] for pyz in cameras
  ], dtype=dtype, device=device)
  camera_pitches = torch.tensor([
    [np.radians(pyz[0])] for pyz in cameras
  ], dtype=dtype, device=device)
  camera_distances = torch.tensor([
    [pow(1.1, pyz[2])] for pyz in cameras
  ], dtype=dtype, device=device)
  viewport = pyrenderer.Camera.viewport_from_sphere(
    camera_center, camera_yaws, camera_pitches, camera_distances, camera_orientation)
  ray_start, ray_dir = pyrenderer.Camera.generate_rays(
    viewport, camera_fov_radians, W, H)

  # tf
  tf_mode = pyrenderer.TFMode.Identity
  opacity_scaling = 2.5
  tf = torch.tensor([[
    # r,g,b,a,pos
    [opacity_scaling, 1]
  ]], dtype=dtype, device=device)

  print("Create renderer inputs")
  inputs = pyrenderer.RendererInputs()
  inputs.screen_size = pyrenderer.int2(W, H)
  #inputs.volume = volume.getDataGpu(0)
  inputs.volume_filter_mode = pyrenderer.VolumeFilterMode.Trilinear
  inputs.box_min = pyrenderer.real3(-0.5, -0.5, -0.5)
  inputs.box_size = pyrenderer.real3(1, 1, 1)
  inputs.camera_mode = pyrenderer.CameraMode.RayStartDir
  inputs.camera = pyrenderer.CameraPerPixelRays(ray_start, ray_dir)
  inputs.step_size = 0.25 / X
  inputs.tf_mode = tf_mode
  inputs.tf = tf
  inputs.blend_mode = pyrenderer.BlendMode.BeerLambert

  # construct autodiff function for the renderer
  class RendererDerivVolume(torch.autograd.Function):
    @staticmethod
    def forward(ctx, current_volume):
      inputs.volume = current_volume
      # allocate outputs
      output_color = torch.empty(B, H, W, 4, dtype=dtype, device=device)
      output_termination_index = torch.empty(B, H, W, dtype=torch.int32, device=device)
      outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)
      # render
      pyrenderer.sync()
      pyrenderer.Renderer.render_forward(inputs, outputs)
      pyrenderer.sync()
      ctx.save_for_backward(current_volume, output_color, output_termination_index)
      return output_color

    @staticmethod
    def backward(ctx, grad_output_color):
      current_volume, output_color, output_termination_index = ctx.saved_tensors
      # reconstruct inputs and outputs
      inputs.volume = current_volume
      outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)
      # construct adjoint outputs
      grad_volume = torch.zeros_like(current_volume)
      gradients_out = pyrenderer.AdjointOutputs()
      gradients_out.has_volume_derivatives = True
      gradients_out.adj_volume = grad_volume
      # adjoint rendering
      pyrenderer.sync()
      pyrenderer.Renderer.render_adjoint(inputs, outputs, grad_output_color, gradients_out)
      pyrenderer.sync()
      return grad_volume
  renderer_deriv_volume = RendererDerivVolume.apply

  # construct the model
  class OptimModel(torch.nn.Module):
    def forward(self, current_volume):
      current_volume = torch.sigmoid(current_volume)
      color = renderer_deriv_volume(current_volume)
      return color, current_volume
  model = OptimModel()
  loss_function = torch.nn.MSELoss()

  # render reference
  print("Render reference")
  reference_color, reference_volume_post = model(reference_volume)

  # render initial
  print("Render initial")
  initial_color, initial_volume_post = model(initial_volume)

  # optimization
  print("Run optimization")
  reconstructed_color = []
  reconstructed_volume_post = []
  reconstructed_loss = []
  current_volume = initial_volume.clone()
  current_volume.requires_grad_()
  optimizer = optimizer_class([current_volume], lr=lr)
  for iteration in range(iterations):
    optimizer.zero_grad()
    color, current_volume_post = model(current_volume)
    loss_value = loss_function(color, reference_color)
    loss_value.backward()

    reconstructed_color.append(color.detach().cpu().numpy())
    reconstructed_volume_post.append(current_volume_post.detach().cpu().numpy())
    reconstructed_loss.append(loss_value.item())
    print("Iteration % 04d, Loss: %.7f" %(iteration, loss_value.item()))

    def closure():
      optimizer.zero_grad()
      color, current_volume_post = model(current_volume)
      loss_value = loss_function(color, reference_color)
      loss_value.backward()
      return loss_value
    optimizer.step(closure=closure)

  # VISUALIZATION
  # columns: (renderings, slices) for (reference, optimization, initial)
  print("Visualize")
  slice_indices = [int((i+0.5)*X/B) for i in range(B)]
  fig, axs = plt.subplots(B, 6, figsize=(18,12))

  # reference
  axs[0, 0].set_title("Reference")
  for i in range(B):
    axs[i, 0].imshow(reference_color.detach().cpu().numpy()[i][:,:,:3])
    axs[i, 1].imshow(
      reference_volume_post.detach().cpu().numpy()[0,slice_indices[i],:,:],
      vmin=0, vmax=1)
  # initial
  axs[0, 4].set_title("Initial")
  for i in range(B):
    axs[i, 4].imshow(initial_color.detach().cpu().numpy()[i][:,:,:3])
    axs[i, 5].imshow(
      initial_volume_post.detach().cpu().numpy()[0, slice_indices[i], :, :],
      vmin=0, vmax=1)

  # reconstruction
  frame = 0
  axs[0, 2].set_title("Reconstruction")
  for i in range(B):
    axs[i, 2].imshow(reconstructed_color[frame][i][:,:,:3])
    axs[i, 3].imshow(
      reconstructed_volume_post[frame][0, slice_indices[i], :, :],
      vmin=0, vmax=1)
  fig.suptitle("Iteration % 4d, Loss: %.7f" % (frame, reconstructed_loss[frame]))

  for i in range(B):
    for j in range(6):
      axs[i, j].set_xticks([])
      axs[i, j].set_yticks([])

  plt.tight_layout()

  if filename is not None:
    print("Write frames")
    with tqdm.tqdm(total=len(reconstructed_color)) as pbar:
      def update(frame=0):
        for i in range(B):
          axs[i, 2].imshow(reconstructed_color[frame][i][:,:,:3])
          axs[i, 3].imshow(
            reconstructed_volume_post[frame][0, slice_indices[i], :, :],
            vmin=0, vmax=1)
        fig.suptitle("Iteration % 4d, Loss: %.7f"%(frame, reconstructed_loss[frame]))
        if frame>0: pbar.update(1)
      update(0); update(1); update(0)
      anim = matplotlib.animation.FuncAnimation(
        fig, update, init_func=update, frames=len(reconstructed_color), blit=False,
        interval=video_length*1000//len(reconstructed_color)) # 10seconds long
      anim.save(filename)

  else:
    plt.show()

  pyrenderer.cleanup()