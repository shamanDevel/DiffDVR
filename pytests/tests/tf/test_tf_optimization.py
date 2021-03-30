import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation
import tqdm

# load pyrenderer
from diffdvr import make_real3
import pyrenderer

from vis import tfvis

# TF parameterization:
# color by Sigmoid, opacity by SoftPlus
class TransformTF(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.sigmoid = torch.nn.Sigmoid()
    self.softplus = torch.nn.Softplus()
  def forward(self, tf):
    assert len(tf.shape)==3
    assert tf.shape[2]==5
    return torch.cat([
      self.sigmoid(tf[:,:,0:3]), #color
      self.softplus(tf[:,:,3:4]), #opacity
      tf[:,:,4:5] # position
      ], dim=2)

class InverseTransformTF(torch.nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, tf):
    def inverseSigmoid(y):
      return torch.log(-y/(y-1))
    def inverseSoftplus(y, beta=1, threshold=20):
      #if y*beta>threshold: return y
      return torch.log(torch.exp(beta*y)-1)/beta
    print(tf.shape)
    assert len(tf.shape) == 3
    assert tf.shape[2] == 5
    return torch.cat([
      inverseSigmoid(tf[:, :, 0:3]),  # color
      inverseSoftplus(tf[:, :, 3:4]),  # opacity
      tf[:, :, 4:5]  # position
    ], dim=2)

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
  dtype = volume.getDataGpu(0).dtype

  # settings
  fov_degree = 45.0
  camera_origin = np.array([0.0, -0.71, -0.70])
  camera_lookat = np.array([0.0, 0.0, 0.0])
  camera_up = np.array([0,-1,0])
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

  invViewMatrix = pyrenderer.Camera.compute_matrix(
    make_real3(camera_origin), make_real3(camera_lookat), make_real3(camera_up),
    fov_degree, W, H)
  print("view matrix:")
  print(np.array(invViewMatrix))

  print("Create renderer inputs")
  inputs = pyrenderer.RendererInputs()
  inputs.screen_size = pyrenderer.int2(W, H)
  inputs.volume = volume.getDataGpu(0)
  inputs.volume_filter_mode = pyrenderer.VolumeFilterMode.Trilinear
  inputs.box_min = pyrenderer.real3(-0.5, -0.5, -0.5)
  inputs.box_size = pyrenderer.real3(1, 1, 1)
  inputs.camera_mode = pyrenderer.CameraMode.InverseViewMatrix
  inputs.camera = invViewMatrix
  inputs.step_size = 0.5 / X
  inputs.tf_mode = tf_mode
  inputs.tf = tf
  inputs.blend_mode = pyrenderer.BlendMode.BeerLambert

  print("Create forward difference settings")
  differences_settings = pyrenderer.ForwardDifferencesSettings()
  differences_settings.D = 4*3 # I want gradients for all inner control points
  derivative_tf_indices = torch.tensor([[
    [-1,-1,-1,-1,-1],
    [0,1,2,3,-1],
    [4,5,6,7,-1],
    [8,9,10,11,-1],
    [-1, -1, -1, -1, -1]
  ]], dtype=torch.int32)
  differences_settings.d_tf = derivative_tf_indices.to(device=device)
  differences_settings.has_tf_derivatives = True

  print("Create renderer outputs")
  output_color = torch.empty(1, H, W, 4, dtype=dtype, device=device)
  output_termination_index = torch.empty(1, H, W, dtype=torch.int32, device=device)
  outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)
  gradients_out = torch.empty(1, H, W, differences_settings.D, 4, dtype=dtype, device=device)

  # render reference
  print("Render reference")
  pyrenderer.Renderer.render_forward(inputs, outputs)
  reference_color_gpu = output_color.clone()
  reference_color_image = output_color.cpu().numpy()[0]
  reference_tf = tf.cpu().numpy()[0]

  # initialize initial TF and render
  print("Render initial")
  initial_tf = torch.tensor([[
    # r,g,b,a,pos
    [0.9,0.01,0.01,0.001,0],
    [0.2, 0.4, 0.3, 10, 0.45],
    [0.6, 0.7, 0.2, 7, 0.5],
    [0.5, 0.6, 0.4, 5, 0.55],
    [0.9,0.99,0.99,0.001,1]
  ]], dtype=dtype, device=device)
  print("Initial tf (original):", initial_tf)
  inputs.tf = initial_tf
  pyrenderer.Renderer.render_forward(inputs, outputs)
  initial_color_image = output_color.cpu().numpy()[0]
  tf = InverseTransformTF()(initial_tf)
  print("Initial tf (transformed):", tf)
  initial_tf = initial_tf.cpu().numpy()[0]

  # Construct the model
  class RendererDerivTF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, current_tf):
      inputs.tf = current_tf
      # render
      pyrenderer.Renderer.render_forward_gradients(inputs, differences_settings, outputs, gradients_out)
      ctx.save_for_backward(current_tf, gradients_out)
      return output_color
    @staticmethod
    def backward(ctx, grad_output_color):
      current_tf, gradients_out = ctx.saved_tensors
      # apply forward derivatives to the adjoint of the color
      # to get the adjoint of the tf
      grad_output_color = grad_output_color.unsqueeze(3) # for broadcasting over the derivatives
      gradients = torch.mul(gradients_out, grad_output_color) # adjoint-multiplication
      gradients = torch.sum(gradients, dim=[1,2,4]) # reduce over screen height, width and channel
      # map to output variables
      grad_tf = torch.zeros_like(current_tf)
      for R in range(grad_tf.shape[1]):
        for C in range(grad_tf.shape[2]):
          idx = derivative_tf_indices[0,R,C]
          if idx>=0:
            grad_tf[:,R,C] = gradients[:,idx]
      return grad_tf
  rendererDerivTF = RendererDerivTF.apply

  class OptimModel(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.tf_transform = TransformTF()
    def forward(self, current_tf):
      # TODO: softplus for opacity, sigmoid for color
      transformed_tf = self.tf_transform(current_tf)
      color = rendererDerivTF(transformed_tf)
      loss = torch.nn.functional.mse_loss(color, reference_color_gpu)
      return loss, transformed_tf, color
  model = OptimModel()

  # run optimization
  iterations = 200
  reconstructed_color = []
  reconstructed_tf = []
  reconstructed_loss = []
  current_tf = tf.clone()
  current_tf.requires_grad_()
  optimizer = torch.optim.Adam([current_tf], lr=0.2)
  for iteration in range(iterations):
    optimizer.zero_grad()
    loss, transformed_tf, color = model(current_tf)
    reconstructed_color.append(color.detach().cpu().numpy()[0,:,:,0:3])
    reconstructed_loss.append(loss.item())
    reconstructed_tf.append(transformed_tf.detach().cpu().numpy()[0])
    loss.backward()
    optimizer.step()
    print("Iteration % 4d, Loss: %7.5f"%(iteration, loss.item()))

  print("Visualize Optimization")
  fig, axs = plt.subplots(3, 2, figsize=(8,6))
  axs[0,0].imshow(reference_color_image[:,:,0:3])
  tfvis.renderTfLinear(reference_tf, axs[0, 1])
  axs[1, 0].imshow(reconstructed_color[0])
  tfvis.renderTfLinear(reconstructed_tf[0], axs[1, 1])
  axs[2,0].imshow(initial_color_image[:,:,0:3])
  tfvis.renderTfLinear(initial_tf, axs[2, 1])
  axs[0,0].set_title("Color")
  axs[0,1].set_title("Transfer Function")
  axs[0,0].set_ylabel("Reference")
  axs[1,0].set_ylabel("Optimization")
  axs[2,0].set_ylabel("Initial")
  for i in range(3):
    for j in range(2):
      axs[i,j].set_xticks([])
      if j==0: axs[i,j].set_yticks([])
  fig.suptitle("Iteration % 4d, Loss: %7.3f" % (0, reconstructed_loss[0]))
  fig.tight_layout()

  print("Write frames")
  with tqdm.tqdm(total=len(reconstructed_color)) as pbar:
    def update(frame):
      axs[1, 0].imshow(reconstructed_color[frame])
      tfvis.renderTfLinear(reconstructed_tf[frame], axs[1, 1])
      fig.suptitle("Iteration % 4d, Loss: %7.5f"%(frame, reconstructed_loss[frame]))
      if frame>0: pbar.update(1)
    anim = matplotlib.animation.FuncAnimation(fig, update, frames=len(reconstructed_color), blit=False)
    anim.save("test_tf_optimization01.mp4")
  plt.show()

  pyrenderer.cleanup()