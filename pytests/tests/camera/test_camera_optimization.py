import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation
import tqdm

# load pyrenderer
from diffdvr import renderer_dtype_torch
import pyrenderer

from vis import lossvis
from vis import cameravis

def make_real3(vector):
  return pyrenderer.real3(vector[0], vector[1], vector[2])

if __name__=='__main__':
  print(pyrenderer.__doc__)

  print("Create Marschner Lobb")
  volume = pyrenderer.Volume.create_implicit(pyrenderer.ImplicitEquation.MarschnerLobb, 64)
  volume.copy_to_gpu()
  print("density tensor: ", volume.getDataGpu(0).shape, volume.getDataGpu(0).dtype, volume.getDataGpu(0).device)

  B = 1 # batch dimension
  H = 256 # screen height
  W = 512 # screen width
  Y = volume.resolution.y
  Z = volume.resolution.z
  X = volume.resolution.x
  device = volume.getDataGpu(0).device
  dtype = renderer_dtype_torch

  # camera settings
  optimize_pitch = True
  optimize_yaw = True
  optimize_distance = True
  optimizer_class = torch.optim.LBFGS
  #optimizer_class = torch.optim.Adam
  filename = "test_camera_optimization-far.gif" # None to disable writing animation
  iterations = 10#20
  lr = 0.1

  fov_radians = np.radians(45.0)
  camera_orientation = pyrenderer.Orientation.Ym
  camera_center = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
  camera_reference_pitch = torch.tensor([[np.radians(-37.5)]], dtype=dtype, device=device)
  camera_reference_yaw = torch.tensor([[np.radians(87.5)]], dtype=dtype, device=device)
  camera_reference_distance = torch.tensor([[0.8]], dtype=dtype, device=device)
  camera_initial_pitch = torch.tensor([[np.radians(30)]], dtype=dtype, device=device) # torch.tensor([[np.radians(-14.5)]], dtype=dtype, device=device)
  camera_initial_yaw = torch.tensor([[np.radians(-20)]], dtype=dtype, device=device) # torch.tensor([[np.radians(113.5)]], dtype=dtype, device=device)
  camera_initial_distance = torch.tensor([[0.7]], dtype=dtype, device=device)

  if not optimize_distance:
    camera_initial_distance = camera_reference_distance
  if not optimize_pitch:
    camera_initial_pitch = camera_reference_pitch
  if not optimize_yaw:
    camera_initial_yaw = camera_reference_yaw

  # TF settings
  opacity_scaling = 25.0
  tf_mode = pyrenderer.TFMode.Linear
  tf = torch.tensor([[
    #r,g,b,a,pos
    [0.9,0.01,0.01,0.001,0],
    [0.9,0.58,0.46,0.001,0.45],
    [0.9,0.61,0.50,0.8*opacity_scaling,0.5],
    [0.9,0.66,0.55,0.001,0.55],
    [0.9,0.99,0.99,0.001,1]
  ]], dtype=dtype, device=device)

  print("Create renderer inputs")
  inputs = pyrenderer.RendererInputs()
  inputs.screen_size = pyrenderer.int2(W, H)
  inputs.volume = volume.getDataGpu(0)
  inputs.volume_filter_mode = pyrenderer.VolumeFilterMode.Trilinear
  inputs.box_min = pyrenderer.real3(-0.5, -0.5, -0.5)
  inputs.box_size = pyrenderer.real3(1, 1, 1)
  inputs.camera_mode = pyrenderer.CameraMode.RayStartDir
  inputs.step_size = 0.25 / X
  inputs.tf_mode = tf_mode
  inputs.tf = tf
  inputs.blend_mode = pyrenderer.BlendMode.BeerLambert

  print("Create forward difference settings")
  differences_settings = pyrenderer.ForwardDifferencesSettings()
  differences_settings.D = 6 # gradients for all camera parameters
  differences_settings.d_rayStart = pyrenderer.int3(0,1,2)
  differences_settings.d_rayDir = pyrenderer.int3(3,4,5)

  # Construct the model
  class RendererDerivCamera(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ray_start, ray_dir):
      inputs.camera = pyrenderer.CameraPerPixelRays(ray_start, ray_dir)
      # allocate outputs
      output_color = torch.empty(1, H, W, 4, dtype=dtype, device=device)
      output_termination_index = torch.empty(1, H, W, dtype=torch.int32, device=device)
      outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)
      gradients_out = torch.empty(1, H, W, differences_settings.D, 4, dtype=dtype, device=device)
      # render
      pyrenderer.Renderer.render_forward_gradients(inputs, differences_settings, outputs, gradients_out)
      ctx.save_for_backward(gradients_out)
      return output_color

    @staticmethod
    def backward(ctx, grad_output_color):
      gradients_out, = ctx.saved_tensors
      # apply forward derivatives to the adjoint of the color
      # to get the adjoint of the tf
      grad_output_color = grad_output_color.unsqueeze(3)  # for broadcasting over the derivatives
      gradients = torch.mul(gradients_out, grad_output_color)  # adjoint-multiplication
      gradients = torch.sum(gradients, dim=4)  # reduce over channel
      # map to output variables
      grad_ray_start = gradients[..., 0:3]
      grad_ray_dir = gradients[..., 3:6]
      return grad_ray_start, grad_ray_dir
  renderer_deriv_camera = RendererDerivCamera.apply

  class OptimModel(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self._loss = torch.nn.MSELoss()
    def forward(self, current_pitch, current_yaw, current_distance, target_color=None):
      #print("pitch:", current_pitch)
      #print("yaw:", current_yaw)
      #print("distance:", current_distance)
      viewport = pyrenderer.Camera.viewport_from_sphere(
        camera_center, current_yaw, current_pitch, current_distance, camera_orientation)
      #print("viewport:", viewport)
      ray_start, ray_dir = pyrenderer.Camera.generate_rays(
        viewport, fov_radians, W, H)
      color = renderer_deriv_camera(ray_start, ray_dir)
      if target_color is None:
        loss = 0
      else:
        loss = self._loss(color, target_color)
      return loss, viewport, color
  model = OptimModel()

  # render reference
  print("Render reference")
  _, reference_viewport, reference_color = model(
    camera_reference_pitch, camera_reference_yaw, camera_reference_distance)

  # render initial
  print("Render reference")
  _, initial_viewport, initial_color = model(
    camera_initial_pitch, camera_initial_yaw, camera_initial_distance)

  # run optimization
  print("Target: pitch=%5.2f, yaw=%5.2f, distance=%5.2f"%(
    camera_reference_pitch.item(), camera_reference_yaw.item(), camera_reference_distance.item()))
  reconstructed_color = []
  reconstructed_viewport = []
  reconstructed_loss = []
  current_pitch = camera_initial_pitch.clone()
  current_yaw = camera_initial_yaw.clone()
  current_distance = camera_initial_distance.clone()
  variables = []
  if optimize_pitch:
    current_pitch.requires_grad_()
    variables.append(current_pitch)
  if optimize_yaw:
    current_yaw.requires_grad_()
    variables.append(current_yaw)
  if optimize_distance:
    current_distance.requires_grad_()
    variables.append(current_distance)
  optimizer = optimizer_class(variables, lr=lr)
  for iteration in range(iterations):
    optimizer.zero_grad()
    loss, current_viewport, color = model(current_pitch, current_yaw, current_distance, reference_color)
    reconstructed_color.append(color.detach().cpu().numpy()[0])
    reconstructed_loss.append(loss.item())
    reconstructed_viewport.append(current_viewport.detach().cpu().numpy()[0])
    loss.backward()
    print("Iteration % 4d, Loss: %7.5f, pitch=%7.4f, yaw=%7.4f, distance=%7.4f" % (
      iteration, loss.item(), current_pitch.item(), current_yaw.item(), current_distance.item()))

    def closure():
      optimizer.zero_grad()
      loss,_,_ = model(current_pitch, current_yaw, current_distance, reference_color)
      loss.backward()
      return loss
    optimizer.step(closure=closure)
  print("Target: pitch=%7.4f, yaw=%7.4f, distance=%7.4f" % (
    camera_reference_pitch.item(), camera_reference_yaw.item(), camera_reference_distance.item()))

  print("Visualize Optimization")
  fig = plt.figure(figsize=(8,6), constrained_layout=True)
  gs = GridSpec(3, 2, figure=fig)
  axes_images = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[2, 0])
  ]
  axes_images[0].imshow(reference_color.cpu().numpy()[0,...,:3])
  axes_images[1].imshow(reconstructed_color[0][...,:3])
  axes_images[2].imshow(initial_color.cpu().numpy()[0,...,:3])

  axes_images[0].set_title("Color")
  axes_images[0].set_ylabel("Reference")
  axes_images[1].set_ylabel("Optimization")
  axes_images[2].set_ylabel("Initial")
  for j in range(3):
    axes_images[j].set_xticks([])
    axes_images[j].set_yticks([])

  axes_camera = fig.add_subplot(gs[0:2,1], projection='3d')
  axes_camera.set_title("Camera")
  initial_viewport = initial_viewport.cpu().numpy()[0]
  reference_viewport = reference_viewport.cpu().numpy()[0]
  reference_sphere_radius = min(camera_initial_distance.item(), camera_reference_distance.item())
  cameravis.renderCamera(
    initial_viewport,
    reference_viewport,
    reconstructed_viewport[0],
    reference_sphere_radius,
    None,
    axes_camera)
  print(axes_camera.elev, axes_camera.azim, axes_camera.dist)
  axes_camera.elev = 20
  axes_camera.dist = 7

  axes_loss = fig.add_subplot(gs[2,1])
  reconstructed_loss = np.array(reconstructed_loss)
  lossvis.renderLoss(reconstructed_loss, axes_loss)

  fig.suptitle("Iteration % 4d, Loss: %7.3f" % (0, reconstructed_loss[0]))
  #fig.tight_layout()

  #print("Initial viewport:")
  #print(initial_viewport.cpu().numpy()[0])
  #print("Target viewport:")
  #print(reference_viewport.cpu().numpy()[0])
  #print("Current viewport:")
  #print(reconstructed_viewport[iterations//2])

  if filename is not None:
    print("Write frames")
    with tqdm.tqdm(total=len(reconstructed_color)) as pbar:
      def update(frame=0):
        axes_images[1].imshow(reconstructed_color[frame][...,:3])

        axes_camera.clear()
        eye_positions = np.array([v[0,:] for v in reconstructed_viewport[:frame+1]])
        cameravis.renderCamera(
          initial_viewport,
          reference_viewport,
          reconstructed_viewport[frame],
          reference_sphere_radius,
          eye_positions,
          axes_camera)

        axes_loss.clear()
        lossvis.renderLoss(reconstructed_loss, axes_loss, frame)

        fig.suptitle("Iteration % 4d, Loss: %7.3f"%(frame, reconstructed_loss[frame]))
        if frame>0: pbar.update(1)
      update(0); update(1); update(0)
      anim = matplotlib.animation.FuncAnimation(
        fig, update, init_func=update, frames=len(reconstructed_color), blit=False,
        interval=4000//len(reconstructed_color)) # 4seconds long
      anim.save(filename)

  else:
    plt.show()

  pyrenderer.cleanup()