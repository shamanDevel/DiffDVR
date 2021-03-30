import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import tqdm
import re
from typing import Optional

from diffdvr import Renderer, CameraOnASphere, Entropy, ColorMatches, Settings, setup_default_settings, \
  fibonacci_sphere, renderer_dtype_torch, renderer_dtype_np
import pyrenderer

def optimize(variables : str, loss_type : str, N : int, settings_file,
             gradient_method : str = "adjoint", forward_delayed_gradients = False,
             show_gradients_in_2d : bool = False,
             optimize_paths : Optional[int] = None, optimizer_class = torch.optim.SGD,
             optimizer_params : dict = None, optimizer_num_iterations = 10,
             optimizer_visualize_intermediate_steps = 10, optimizer_show_arrows=False,
             enable_profiler : bool = False):
  assert variables in ["yaw", "yaw+pitch"]
  assert loss_type in ["entropy", "image"]
  assert N > 0
  assert gradient_method in ["adjoint", "forward"]

  np.random.seed(42)
  torch.random.manual_seed(42)

  # settings
  run_on_cuda = True
  H = 512  # screen height
  W = 512  # screen width

  s = Settings(settings_file)
  volume = s.load_dataset()
  volume.copy_to_gpu()
  volume_data = volume.getDataGpu(0) if run_on_cuda else volume.getDataCpu(0)
  device = volume_data.device
  dtype = renderer_dtype_torch
  dtype_np = renderer_dtype_np

  rs = setup_default_settings(
    volume, W, H, s.get_stepsize(), run_on_cuda)
  renderer = Renderer(rs, optimize_camera=True,
                      gradient_method=gradient_method,
                      forward_delayed_gradients=forward_delayed_gradients)

  # TF
  renderer.settings.tf_mode = pyrenderer.TFMode.Linear
  tf_points = s.get_tf_points()
  tf = pyrenderer.TFUtils.get_piecewise_tensor(tf_points)
  tf = tf.to(dtype=dtype, device=device)
  peak_colors = []
  for p in tf_points:
    if p.val.w > 0:
      peak_colors.append([p.val.x, p.val.y, p.val.z])
  peak_colors = np.array(peak_colors)

  # camera
  # for a first test, sample N points along the yaw axis
  # camera_yaw traces gradients
  Batches = min(N, 10)
  assert N % Batches == 0

  camera_config = s.get_camera()
  if variables=="yaw":
    camera_yaw_cpu = np.linspace(0, 2*np.pi, N, endpoint=False, dtype=dtype_np)
    camera_pitch_cpu = camera_config.pitch_radians * np.ones((N,), dtype=dtype_np)
  elif variables=="yaw+pitch":
    camera_pitch_cpu, camera_yaw_cpu = fibonacci_sphere(N, dtype=dtype_np)
    print("pitch:\n", camera_pitch_cpu)
    print("yaw:\n", camera_yaw_cpu)
  camera_distance_cpu = camera_config.distance * np.ones((N,), dtype=dtype_np)
  camera_center_cpu = np.zeros((N, 3), dtype=dtype_np)
  camera_fov_radians = camera_config.fov_y_radians
  camera_module = CameraOnASphere(camera_config.orientation)

  # allocate outputs
  loss_values_cpu = np.zeros((N,), dtype=dtype_np)
  grad_yaw_cpu = np.zeros((N,), dtype=dtype_np)
  grad_pitch_cpu = np.zeros((N,), dtype=dtype_np)
  cameras_cpu = np.zeros((N,3,3), dtype=dtype_np)
  images_cpu = np.zeros((N,H,W,4), dtype=dtype_np)

  # define loss
  if loss_type == "entropy":
    class EntropyLosses(torch.nn.Module):
      def __init__(self):
        super().__init__()

        self._opacity_entropy = Entropy(dim=(1,2), normalize_input=True, normalize_output=True)
        self._opacity_weight = 1

        self._color_matches = ColorMatches(peak_colors)
        self._color_entropy = Entropy(dim=(1,), normalize_input=False)
        self._color_weight = 0

      def forward(self, x):
        losses = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

        if self._opacity_weight > 0:
          opacity = x[:,:,:,3]
          opacity_entropy = self._opacity_entropy(opacity)
          assert losses.shape == opacity_entropy.shape
          losses = losses + self._opacity_weight * opacity_entropy

        if self._color_weight > 0:
          color_regions = self._color_matches(x)
          color_entropy = self._color_entropy(color_regions)
          assert losses.shape == color_entropy.shape
          losses = losses + self._color_weight * color_entropy

        # TODO: measure of the stability
        # maybe some form of summed per-pixel gradients for yaw+pitch
        # how does this work in an adjoint framework?

        return losses

    loss = EntropyLosses()

  elif loss_type == "image":
    print("Render reference image")
    reference_index = np.random.randint(0, N)
    def render_single_image(index):
      istart = index; iend = index+1

      # camera
      camera_yaw = torch.from_numpy(camera_yaw_cpu[istart:iend]).to(device=device).unsqueeze(1)
      camera_pitch = torch.from_numpy(camera_pitch_cpu[istart:iend]).to(device=device).unsqueeze(1)
      camera_distance = torch.from_numpy(camera_distance_cpu[istart:iend]).to(device=device).unsqueeze(1)
      camera_center = torch.from_numpy(camera_center_cpu[istart:iend]).to(device=device)
      cameras = camera_module(camera_center, camera_yaw, camera_pitch, camera_distance)

      # render
      images = renderer(
        camera=cameras, fov_y_radians=camera_fov_radians,
        tf=tf, volume=volume_data)
      return images.detach()
    reference_image = render_single_image(reference_index)

    class ImageLoss(torch.nn.Module):
      def __init__(self, reference):
        super().__init__()
        self.register_buffer("_reference", reference)
        self._loss = torch.nn.MSELoss(reduction='none')
      def forward(self, x):
        return -torch.mean(self._loss(x, self._reference), dim=(1,2,3))

    loss = ImageLoss(reference_image)

  loss.to(device=device, dtype=dtype)

  # render
  if enable_profiler:
    event_list = torch.autograd.profiler.EventList(use_cuda=run_on_cuda, profile_memory=True)
  for b in tqdm.trange(N//Batches):
    istart = b*Batches; iend = istart+Batches

    # camera
    camera_yaw = torch.from_numpy(camera_yaw_cpu[istart:iend]).to(device=device).unsqueeze(1)
    camera_pitch = torch.from_numpy(camera_pitch_cpu[istart:iend]).to(device=device).unsqueeze(1)
    camera_distance = torch.from_numpy(camera_distance_cpu[istart:iend]).to(device=device).unsqueeze(1)
    camera_center = torch.from_numpy(camera_center_cpu[istart:iend]).to(device=device)
    if variables=="yaw":
      camera_yaw.requires_grad_(True)
    elif variables=="yaw+pitch":
      camera_yaw.requires_grad_(True)
      camera_pitch.requires_grad_(True)
    cameras = camera_module(camera_center, camera_yaw, camera_pitch, camera_distance)
    cameras_cpu[istart:iend, ...] = cameras.detach().cpu().numpy()

    with torch.autograd.profiler.profile(
        enabled = enable_profiler, use_cuda=run_on_cuda,
        record_shapes=True, profile_memory=True) as prof:
      # render
      images = renderer(
        camera=cameras, fov_y_radians=camera_fov_radians,
        tf=tf, volume=volume_data)

      # compute entropy
      loss_values = loss(images)

      # Backward
      torch.sum(loss_values).backward()

    # copy to CPU
    images_cpu[istart:iend, ...] = images.detach().cpu().numpy()
    loss_values_cpu[istart:iend, ...] = loss_values.detach().cpu().numpy()
    if variables=="yaw":
      grad_yaw_cpu[istart:iend,...] = camera_yaw.grad[:, 0].detach().cpu().numpy()
    elif variables=="yaw+pitch":
      grad_yaw_cpu[istart:iend,...] = camera_yaw.grad[:, 0].detach().cpu().numpy()
      grad_pitch_cpu[istart:iend,...] = camera_pitch.grad[:, 0].detach().cpu().numpy()

    # accumulate profiling results
    if enable_profiler:
      event_list += prof.function_events

  best_index = np.argmax(loss_values_cpu)
  worst_index = np.argmin(loss_values_cpu)
  if enable_profiler:
    #filter keys
    name_filter = re.compile("_RendererFunction")
    averaged = event_list.key_averages()
    filtered = torch.autograd.profiler.EventList(
      [event for event in averaged if name_filter.match(event.key)],
      use_cuda=run_on_cuda, profile_memory=True)
    print(filtered.table())

  # finite-differences gradients
  if variables=="yaw":
    def compute_grad_fd(v):
      n = v.shape[0]
      v2 = np.concatenate([v,v,v])
      return (v2[n+1:2*n+1] - v2[n-1:2*n-1]) / 2
    grad_yaw_fd = compute_grad_fd(loss_values_cpu) * (N/(2*np.pi))
    #grad_yaw_fd = grad_yaw_fd / np.max(grad_yaw_fd) * np.max(grad_yaw_cpu)

  # optimizatoin
  if optimize_paths is not None:
    print("Optimize 8 paths")
    M = optimize_paths

    #camera
    if variables == "yaw":
      initial_camera_yaw_cpu = np.linspace(0.2*np.pi, 2.2 * np.pi, M, endpoint=False, dtype=dtype_np)
      initial_camera_pitch_cpu = camera_config.pitch_radians * np.ones((M,), dtype=dtype_np)
    elif variables == "yaw+pitch":
      M = min(8, optimize_paths)
      initial_camera_pitch_cpu = np.array([
        np.deg2rad(45), np.deg2rad(45), np.deg2rad(45), np.deg2rad(45),
        np.deg2rad(-45), np.deg2rad(-45), np.deg2rad(-45), np.deg2rad(-45)
      ], dtype=dtype_np)[:M]
      initial_camera_yaw_cpu = np.array([
        np.deg2rad(45), np.deg2rad(135), np.deg2rad(225), np.deg2rad(315),
        np.deg2rad(45), np.deg2rad(135), np.deg2rad(225), np.deg2rad(315)
      ], dtype=dtype_np)[:M]
      print("pitch:\n", camera_pitch_cpu)
      print("yaw:\n", camera_yaw_cpu)
    initial_camera_distance_cpu = camera_config.distance * np.ones((M,), dtype=dtype_np)
    initial_camera_center_cpu = np.zeros((M, 3), dtype=dtype_np)
    initial_camera_fov_radians = camera_config.fov_y_radians
    # copy to GPU
    current_camera_yaw = torch.from_numpy(initial_camera_yaw_cpu).to(device=device).unsqueeze(1)
    current_camera_pitch = torch.from_numpy(initial_camera_pitch_cpu).to(device=device).unsqueeze(1)
    current_camera_distance = torch.from_numpy(initial_camera_distance_cpu).to(device=device).unsqueeze(1)
    current_camera_center = torch.from_numpy(initial_camera_center_cpu).to(device=device)
    optimization_parameters = []
    if variables == "yaw":
      current_camera_yaw.requires_grad_(True)
      optimization_parameters = [current_camera_yaw]
    elif variables == "yaw+pitch":
      current_camera_yaw.requires_grad_(True)
      current_camera_pitch.requires_grad_(True)
      optimization_parameters = [current_camera_yaw, current_camera_pitch]
    current_cameras = camera_module(current_camera_center, current_camera_yaw,
                                    current_camera_pitch, current_camera_distance)
    current_cameras_cpu = current_cameras.detach().cpu().numpy()
    # create optimizer
    if optimizer_params is None:
      optimizer_params = {'lr':1.0}
    optimizer = optimizer_class(optimization_parameters, **optimizer_params)
    # create optimization closure
    last_loss = 0
    def optim_closure():
      nonlocal last_loss
      optimizer.zero_grad()
      cam = camera_module(current_camera_center, current_camera_yaw,
                          current_camera_pitch, current_camera_distance)
      images = renderer(
        camera=cam, fov_y_radians=camera_fov_radians,
        tf=tf, volume=volume_data)
      loss_values = loss(images)
      loss_value = torch.sum(loss_values)
      loss_value.backward()
      last_loss = loss_value.item()
      return loss_value

    # allocate output
    optimized_positions = [current_cameras_cpu[:,0,:]] # B*xyz
    optimized_yaws = [initial_camera_yaw_cpu.copy()]
    optimized_intermediate_positions = [current_cameras_cpu[:,0,:]]
    previous_camera_yaw = current_camera_yaw.detach().clone()
    previous_camera_pitch = current_camera_pitch.detach().clone()
    loss_per_iteration = [optim_closure().item()]

    # optimize
    with tqdm.tqdm(optimizer_num_iterations+1) as iteration_bar:
      for iteration in range(optimizer_num_iterations+1):
        optimizer.step(optim_closure)
        with torch.no_grad():
          current_cameras = camera_module(current_camera_center, current_camera_yaw,
                                          current_camera_pitch, current_camera_distance)
          current_cameras_cpu = current_cameras.detach().cpu().numpy()
          optimized_positions.append(current_cameras_cpu[:,0,:])
          optimized_yaws.append(current_camera_yaw.detach().cpu().numpy()[:,0])
          for i in range(optimizer_visualize_intermediate_steps):
            frac = (i+1) / (optimizer_visualize_intermediate_steps+2)
            intermediate_camera_yaw = (1-frac)*previous_camera_yaw + frac*current_camera_yaw
            intermediate_camera_pitch = (1 - frac) * previous_camera_pitch + frac * current_camera_pitch
            intermediate_cameras_cpu = camera_module(current_camera_center, intermediate_camera_yaw,
                                            intermediate_camera_pitch, current_camera_distance).cpu().numpy()
            optimized_intermediate_positions.append(intermediate_cameras_cpu[:, 0, :])
          optimized_intermediate_positions.append(current_cameras_cpu[:, 0, :])
          previous_camera_yaw = current_camera_yaw.detach().clone()
          previous_camera_pitch = current_camera_pitch.detach().clone()
        iteration_bar.update(1)
        iteration_bar.set_description("Loss: %5.3f"%last_loss)
        loss_per_iteration.append(last_loss)
    print("Done")
    print("Losses:", loss_per_iteration)
    print("Yaws:", optimized_yaws)

  print("Visualize")
  # TEST: visualize images
  if False:
    fig, axs = plt.subplots(1, N,
                            figsize=(2 * N, 2.5),
                            squeeze=True)
    fig.suptitle("Viewports")
    for i in range(N):
      axs[i].imshow(images_cpu[i,:,:,:3])
      axs[i].set_xticks([])
      axs[i].set_yticks([])
    plt.tight_layout()
    plt.show()

  # TEST: show plot + best and worst
  fig = plt.figure(figsize=(10,6))

  if variables=="yaw":
    axE = plt.subplot(211)
    color = 'tab:blue'
    axE.set_xlabel("Yaw")
    if loss_type == "entropy":
      axE.set_ylabel("Entropy", color=color)
    elif loss_type == "image":
      axE.set_ylabel("Image score", color=color)
    axE.plot(camera_yaw_cpu, loss_values_cpu, color=color)
    axE.tick_params(axis='y', labelcolor=color)
    axE2 = axE.twinx()
    color = 'tab:red'
    axE2.set_ylabel("Gradient", color=color)
    axE2.plot(camera_yaw_cpu, grad_yaw_cpu, color=color, label="analytic")
    axE2.plot(camera_yaw_cpu, grad_yaw_fd, color='orange', linestyle=":", label="finite differences")
    axE2.axhline(0.0, color=color, linestyle="--", alpha=0.5)
    axE2.tick_params(axis='y', labelcolor=color)
    if loss_type == "image":
      axE.axvline(camera_yaw_cpu[reference_index], linestyle=':')
    axE2.legend()

    if optimize_paths is not None:
      M = optimized_yaws[0].shape[0]
      for m in range(M):
        X = np.mod(np.array([yaw[m] for yaw in optimized_yaws]), 2*np.pi)
        Y = np.interp(X, camera_yaw_cpu, loss_values_cpu, period=2*np.pi)
        print("X:", X)
        print("Y:", Y)
        line = axE.plot(X, Y, linestyle='--', color='k')[0]
        if optimizer_show_arrows:
          for i in range(1, len(X)):
            axE.annotate('',
              xytext=(X[i-1], Y[i-1]), xy=(0.5*(X[i]+X[i-1]), 0.5*(Y[i]+Y[i-1])),
              arrowprops=dict(arrowstyle="->", color=line.get_color()),
              size=15)
            #axE.arrow(0.5*(X[i]+X[i-1]), 0.5*(Y[i]+Y[i-1]), 0.005*(X[i]-X[i-1]), 0.005*(Y[i]-Y[i-1]),
            #          color=line.get_color(), width=0, head_width=0.002, head_length=0.05)

  elif variables == "yaw+pitch":
    axN = plt.subplot(221)
    axS = plt.subplot(222)
    axN.set_title("North")
    axS.set_title("South")
    positions = cameras_cpu[:,0,:]
    min_cost = np.min(loss_values_cpu)
    max_cost = np.max(loss_values_cpu)
    norm = matplotlib.colors.Normalize(vmin=min_cost, vmax=max_cost)

    if show_gradients_in_2d:
      # compute gradient for each position
      grad_stepsize = -2
      next_camera_yaw = torch.from_numpy(camera_yaw_cpu+grad_stepsize*grad_yaw_cpu).unsqueeze(1)
      next_camera_pitch = torch.from_numpy(camera_pitch_cpu+grad_stepsize*grad_pitch_cpu).unsqueeze(1)
      next_camera_distance = torch.from_numpy(camera_distance_cpu).unsqueeze(1)
      next_camera_center = torch.from_numpy(camera_center_cpu)
      next_cameras_cpu = camera_module(next_camera_center, next_camera_yaw, next_camera_pitch, next_camera_distance).numpy()
      next_positions = next_cameras_cpu[:,0,:]

    for i,ax in enumerate([axN, axS]):
      # filter by z
      if i==0:
        mask = positions[:,2] >= 0
      else:
        mask = positions[:,2] < 0
      x = positions[mask,0]
      y = positions[mask,1]
      z = loss_values_cpu[mask]
      #patch = patches.Circle((0,0), radius=1, transform=ax.transData)
      ridges = ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k', norm=norm)
      cntr = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r", norm=norm)
      #ridges.set_clip_path(patch)
      #cntr.set_clip_path(path)
      if i==1:
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=ax)
      ax.plot(x, y, 'ko', ms=2)
      if loss_type == "image" and mask[reference_index]:
        ax.plot([positions[reference_index,0]], [positions[reference_index,1]], 'ko', ms=5)
      ax.set_aspect('equal')
      ax.set_xlim(-camera_config.distance, +camera_config.distance)
      ax.set_ylim(-camera_config.distance, +camera_config.distance)

      if show_gradients_in_2d:
        x2 = next_positions[mask, 0]
        y2 = next_positions[mask, 1]
        for (x_start, y_start, x_end, y_end) in zip(x,y,x2,y2):
          ax.annotate('', xytext=(x_start, y_start), xy=(x_end, y_end),
                      arrowprops=dict(arrowstyle="->", color='k', alpha=0.5),
                      size=15)

      if optimize_paths is not None:
        M = optimized_intermediate_positions[0].shape[0]
        for m in range(M):
          local_positions1 = np.array([p[m,:] for p in optimized_intermediate_positions])
          local_positions2 = np.array([p[m, :] for p in optimized_positions])
          if i==0:
            mask1 = local_positions1[:, 2] >= 0
            mask2 = local_positions2[:, 2] >= 0
          else:
            mask1 = local_positions1[:, 2] < 0
            mask2 = local_positions2[:, 2] < 0
          x1 = local_positions1[mask1, 0]
          y1 = local_positions1[mask1, 1]
          x2 = local_positions2[mask2, 0]
          y2 = local_positions2[mask2, 1]
          line = ax.plot(x1,y1,ms=1)[0]
          if optimizer_show_arrows:
            for j in range(1, len(x2), optimizer_show_arrows if isinstance(optimizer_show_arrows, int) else 1):
              pos_end = (x2[j], y2[j])
              pos_start = (0.1*x2[j-1]+0.9*x2[j], 0.1*y2[j-1]+0.9*y2[j])
              ax.annotate('', xytext=pos_start, xy=pos_end,
                          arrowprops=dict(arrowstyle="->", color=line.get_color()),
                          size=15)


  axBest = plt.subplot(223)
  axBest.set_title("Best view, yaw=%.3f, pitch=%.3f"%(
    camera_yaw_cpu[best_index], camera_pitch_cpu[best_index]))
  axBest.imshow(images_cpu[best_index,:,:,:3])
  axBest.set_xticks([])
  axBest.set_yticks([])

  axWorst = plt.subplot(224)
  axWorst.set_title("Worst view, yaw=%.3f, pitch=%.3f"%(
    camera_yaw_cpu[worst_index], camera_pitch_cpu[worst_index]))
  axWorst.imshow(images_cpu[worst_index, :, :, :3])
  axWorst.set_xticks([])
  axWorst.set_yticks([])

  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  #optimize("yaw", "entropy", 60, "config-files/skull1.json")
  #optimize("yaw", "image", 60, "config-files/skull1.json")
  #optimize("yaw+pitch", "image", 180, "config-files/skull1.json")

  #optimize("yaw", "entropy", 60, "config-files/tooth1.json", "adjoint", enable_profiler=True)
  #optimize("yaw", "entropy", 60, "config-files/tooth1.json", "forward", False, enable_profiler=True)
  #optimize("yaw", "entropy", 60, "config-files/tooth1.json", "forward", True, enable_profiler=True)
  #optimize("yaw+pitch", "entropy", 360, "config-files/tooth1.json")

  #optimize("yaw+pitch", "entropy", 60, "config-files/tooth1.json",
  #         optimize_paths=1, optimizer_class=torch.optim.LBFGS, optimizer_params={'lr':1.0},
  #         optimizer_num_iterations=10)
  optimize("yaw+pitch", "image", 30, "../../config-files/tooth1.json",
           show_gradients_in_2d = True,
           optimize_paths=8, optimizer_class=torch.optim.SGD,
           optimizer_params={'lr': 0.4, 'momentum':0.5, 'dampening':0.1},
           optimizer_num_iterations=20, optimizer_show_arrows=10)
  #optimize("yaw+pitch", "image", 360, "config-files/tooth1.json",
  #         show_gradients_in_2d=True,
  #         optimize_paths=8, optimizer_class=torch.optim.LBFGS,
  #         optimizer_params={'lr': 0.1, 'max_iter': 5},
  #         optimizer_num_iterations=20, optimizer_show_arrows=False,
  #         optimizer_visualize_intermediate_steps=50)
