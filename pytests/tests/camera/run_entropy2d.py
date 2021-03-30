import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mticker
import matplotlib
import matplotlib.colors
import tqdm
import imageio

from diffdvr import Renderer, CameraOnASphere, Entropy, ColorMatches, Settings, setup_default_settings, \
    fibonacci_sphere, renderer_dtype_torch, renderer_dtype_np, ProfileRenderer, cvector_to_numpy
from results import get_result_folder
from tests import CommonVariables
import pyrenderer


def compute(settings_file, name, include_color_in_entropy=False, save=True, params=None):
    if params is None:
        params = {}
    N_background = 1024  # number of points in the background
    N_foreground = 256  # number of points in the foreground with gradients
    N_optim = 8  # optimization steps
    Batches = 8

    #optimizer_class = torch.optim.SGD
    #optimizer_params = {'lr': 0.8, 'momentum': 0.5, 'dampening': 0.1}
    #optimizer_num_iterations = 100
    optimizer_class = torch.optim.SGD
    optimizer_params = {'lr': params.get('lr', 5.0)}
    if 'momentum' in params: optimizer_params['momentum'] = params['momentum']
    if 'dampening' in params: optimizer_params['dampening'] = params['dampening']
    optimizer_num_iterations = params.get('iterations', 20)
    optimizer_visualize_intermediate_steps = params.get('intermediate', 50)

    # settings
    s = Settings(settings_file)
    volume = s.load_dataset()
    volume.copy_to_gpu()
    volume_data = volume.getDataGpu(0) if CommonVariables.run_on_cuda else volume.getDataCpu(0)
    device = volume_data.device
    camera_config = s.get_camera()
    H = CommonVariables.screen_height
    W = CommonVariables.screen_width

    rs = setup_default_settings(
        volume, CommonVariables.screen_width, CommonVariables.screen_height,
        s.get_stepsize(), CommonVariables.run_on_cuda)
    rs.tf_mode = pyrenderer.TFMode.Linear
    tf_points = s.get_tf_points()
    tf = pyrenderer.TFUtils.get_piecewise_tensor(tf_points)
    tf = tf.to(dtype=renderer_dtype_torch, device=device)
    # extract peak colors
    peak_colors = []
    for i in range(1, len(tf_points)-1):
        if tf_points[i].val.w > tf_points[i - 1].val.w and \
                tf_points[i].val.w > tf_points[i + 1].val.w:
            peak_colors.append(cvector_to_numpy(tf_points[i].val))
    if len(peak_colors)>0:
        peak_colors = np.stack(peak_colors)
    else:
        if include_color_in_entropy:
            print("No peaks in TF found, disable color in entropy")
        include_color_in_entropy = False

    renderer = {
        'adjoint': Renderer(rs, optimize_camera=True,
                            gradient_method='adjoint'),
        'forward-immediate': Renderer(rs, optimize_camera=True,
                                      gradient_method='forward',
                                      forward_delayed_gradients=False),
        'forward-delayed': Renderer(rs, optimize_camera=True,
                                    gradient_method='forward',
                                    forward_delayed_gradients=True),
    }

    # loss
    class EntropyLosses(torch.nn.Module):
        def __init__(self, opacity_weight=1, color_weight=0):
            super().__init__()

            self._opacity_entropy = Entropy(dim=(1, 2), normalize_input=True, normalize_output=True)
            self._opacity_weight = opacity_weight

            if color_weight>0:
                self._color_matches = ColorMatches(peak_colors)
                self._color_entropy = Entropy(dim=(1,), normalize_input=False)
            self._color_weight = color_weight

        def forward(self, x):
            losses = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

            if self._opacity_weight > 0:
                opacity = x[:, :, :, 3]
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

    loss = EntropyLosses(color_weight=1 if include_color_in_entropy else 0)
    loss.to(device)

    def renderSphere(renderer: Renderer, N: int,
                     evaluate_gradient: bool, profile: bool,
                     save_images: bool):
        camera_pitch_cpu, camera_yaw_cpu = fibonacci_sphere(N, dtype=renderer_dtype_np)
        camera_distance_cpu = camera_config.distance * np.ones((N,), dtype=renderer_dtype_np)
        camera_center_cpu = np.stack([camera_config.center] * N, axis=0).astype(dtype=renderer_dtype_np)
        camera_fov_radians = camera_config.fov_y_radians
        camera_module = CameraOnASphere(camera_config.orientation)

        # allocate outputs
        loss_values_cpu = np.zeros((N,), dtype=renderer_dtype_np)
        grad_yaw_cpu = np.zeros((N,), dtype=renderer_dtype_np)
        grad_pitch_cpu = np.zeros((N,), dtype=renderer_dtype_np)
        cameras_cpu = np.zeros((N, 3, 3), dtype=renderer_dtype_np)
        images_cpu = np.zeros((N, H, W, 4), dtype=renderer_dtype_np) if save_images else None

        forward_milliseconds = np.zeros((N//Batches, ), dtype=renderer_dtype_np)
        backward_milliseconds = np.zeros((N // Batches,), dtype=renderer_dtype_np)
        forward_memory = np.zeros((N // Batches,), dtype=renderer_dtype_np)

        for b in tqdm.trange(N // Batches):
            istart = b * Batches
            iend = istart + Batches

            # camera
            camera_yaw = torch.from_numpy(camera_yaw_cpu[istart:iend]).to(device=device).unsqueeze(1)
            camera_pitch = torch.from_numpy(camera_pitch_cpu[istart:iend]).to(device=device).unsqueeze(1)
            camera_distance = torch.from_numpy(camera_distance_cpu[istart:iend]).to(device=device).unsqueeze(1)
            camera_center = torch.from_numpy(camera_center_cpu[istart:iend]).to(device=device)
            if evaluate_gradient:
                camera_yaw.requires_grad_(True)
                camera_pitch.requires_grad_(True)
            cameras = camera_module(camera_center, camera_yaw, camera_pitch, camera_distance)
            cameras_cpu[istart:iend, ...] = cameras.detach().cpu().numpy()

            profiling = ProfileRenderer() if profile else None
            images = renderer(
                camera=cameras, fov_y_radians=camera_fov_radians,
                tf=tf, volume=volume_data, profiling=profiling)

            loss_values = loss(images)

            if evaluate_gradient:
                loss_scalar = torch.sum(loss_values)
                loss_scalar.backward()

            if profile:
                print("Timings: forward=%.4f, backward=%.4f"%(profiling.forward_ms, profiling.backward_ms))
                forward_milliseconds[b] = profiling.forward_ms / Batches
                backward_milliseconds[b] = profiling.backward_ms / Batches
                forward_memory[b] = profiling.forward_bytes / Batches

            loss_values_cpu[istart:iend, ...] = loss_values.detach().cpu().numpy()
            if save_images:
                images_cpu[istart:iend, ...] = images.detach().cpu().numpy()
            if evaluate_gradient:
                grad_yaw_cpu[istart:iend, ...] = camera_yaw.grad[:, 0].detach().cpu().numpy()
                grad_pitch_cpu[istart:iend, ...] = camera_pitch.grad[:, 0].detach().cpu().numpy()

        return {
            "camera_orientation": camera_config.orientation,
            "camera_pitch": camera_pitch_cpu,
            "camera_yaw": camera_yaw_cpu,
            "camera_distance": camera_distance_cpu,
            "camera_center": camera_center_cpu,
            "camera_fov_radians": camera_fov_radians,
            "loss_values": loss_values_cpu,
            "grad_yaw": grad_yaw_cpu,
            "grad_pitch": grad_pitch_cpu,
            "cameras": cameras_cpu,
            "images": images_cpu,
            "forward_times_ms": forward_milliseconds,
            "backward_times_ms": backward_milliseconds,
            "forward_memory_bytes": forward_memory
        }

    full_results = {}

    print("Render background images")
    background = renderSphere(renderer['forward-delayed'], N_background,
                              False, False, False)
    full_results['background'] = background

    full_results_for_comparison = dict()
    for renderer_key, renderer_instance in renderer.items():
        print("Render foreground with gradients:", renderer_key)
        foreground = renderSphere(renderer_instance, N_foreground,
                                  True, True, renderer_key == 'forward-delayed')
        full_results_for_comparison[renderer_key] = foreground
        if renderer_key == 'forward-delayed':
            full_results['forward-delayed'] = foreground
        else:
            # save only timings
            full_results[renderer_key] = {
                'forward_times_ms': foreground['forward_times_ms'],
                'backward_times_ms': foreground['backward_times_ms'],
                'forward_memory_bytes': foreground['forward_memory_bytes']
            }

    # compare gradients
    for key1 in renderer.keys():
        for key2 in [k for k in renderer.keys() if k!=key1]:
            diff_yaw = np.sum((full_results_for_comparison[key1]['grad_yaw']-
                               full_results_for_comparison[key2]['grad_yaw'])**2)
            diff_pitch = np.sum((full_results_for_comparison[key1]['grad_pitch'] -
                               full_results_for_comparison[key2]['grad_pitch']) ** 2)
            print("Difference between '%s' and '%s': yaw: %.5f, pitch: %.5f"%(
                key1, key2, diff_yaw, diff_pitch))

    def optimize():
        M = min(8, N_optim)
        initial_camera_pitch_cpu = np.array([
            np.deg2rad(45), np.deg2rad(45), np.deg2rad(45), np.deg2rad(45),
            np.deg2rad(-45), np.deg2rad(-45), np.deg2rad(-45), np.deg2rad(-45)
        ], dtype=renderer_dtype_np)[:M]
        initial_camera_yaw_cpu = np.array([
            np.deg2rad(45), np.deg2rad(135), np.deg2rad(225), np.deg2rad(315),
            np.deg2rad(45), np.deg2rad(135), np.deg2rad(225), np.deg2rad(315)
        ], dtype=renderer_dtype_np)[:M]
        initial_camera_distance_cpu = camera_config.distance * np.ones((M,), dtype=renderer_dtype_np)
        initial_camera_center_cpu = np.stack([camera_config.center] * M, axis=0).astype(dtype=renderer_dtype_np)
        initial_camera_fov_radians = camera_config.fov_y_radians
        camera_module = CameraOnASphere(camera_config.orientation)

        current_camera_yaw = torch.from_numpy(initial_camera_yaw_cpu).to(device=device).unsqueeze(1)
        current_camera_pitch = torch.from_numpy(initial_camera_pitch_cpu).to(device=device).unsqueeze(1)
        current_camera_distance = torch.from_numpy(initial_camera_distance_cpu).to(device=device).unsqueeze(1)
        current_camera_center = torch.from_numpy(initial_camera_center_cpu).to(device=device)
        current_camera_yaw.requires_grad_(True)
        current_camera_pitch.requires_grad_(True)
        optimization_parameters = [current_camera_yaw, current_camera_pitch]
        current_cameras = camera_module(current_camera_center, current_camera_yaw,
                                        current_camera_pitch, current_camera_distance)
        current_cameras_cpu = current_cameras.detach().cpu().numpy()

        optimizer = optimizer_class(optimization_parameters, **optimizer_params)
        last_loss = None
        last_images = None

        def optim_closure():
            nonlocal last_loss, last_images
            optimizer.zero_grad()
            cam = camera_module(current_camera_center, current_camera_yaw,
                                current_camera_pitch, current_camera_distance)
            #images = renderer['forward-immediate'](
            images = renderer['adjoint'](
                camera=cam, fov_y_radians=initial_camera_fov_radians,
                tf=tf, volume=volume_data)
            loss_values = loss(images)
            loss_value = -torch.sum(loss_values)
            loss_value.backward()
            last_loss = loss_values.detach().cpu().numpy()
            last_images = images.detach().cpu().numpy()
            return loss_value

        # allocate output
        optimized_positions = [current_cameras_cpu[:, :, :]]  # B*xyz
        optimized_intermediate_positions = [current_cameras_cpu[:, 0, :]]
        previous_camera_yaw = current_camera_yaw.detach().clone()
        previous_camera_pitch = current_camera_pitch.detach().clone()
        optim_closure()
        loss_per_iteration = [last_loss]
        initial_images_cpu = renderer['forward-immediate'](
                camera=camera_module(current_camera_center, current_camera_yaw,
                                current_camera_pitch, current_camera_distance),
                fov_y_radians=initial_camera_fov_radians,
                tf=tf, volume=volume_data).detach().cpu().numpy()
        intermediate_images_cpu = []

        # optimize
        with tqdm.tqdm(optimizer_num_iterations + 1) as iteration_bar:
            for iteration in range(optimizer_num_iterations + 1):
                optimizer.step(optim_closure)
                with torch.no_grad():
                    current_cameras = camera_module(current_camera_center, current_camera_yaw,
                                                    current_camera_pitch, current_camera_distance)
                    current_cameras_cpu = current_cameras.detach().cpu().numpy()
                    optimized_positions.append(current_cameras_cpu[:, :, :])
                    for i in range(optimizer_visualize_intermediate_steps):
                        frac = (i + 1) / (optimizer_visualize_intermediate_steps + 2)
                        intermediate_camera_yaw = (1 - frac) * previous_camera_yaw + frac * current_camera_yaw
                        intermediate_camera_pitch = (1 - frac) * previous_camera_pitch + frac * current_camera_pitch
                        intermediate_cameras_cpu = camera_module(current_camera_center, intermediate_camera_yaw,
                                                                 intermediate_camera_pitch,
                                                                 current_camera_distance).cpu().numpy()
                        optimized_intermediate_positions.append(intermediate_cameras_cpu[:, 0, :])
                    optimized_intermediate_positions.append(current_cameras_cpu[:, 0, :])
                    previous_camera_yaw = current_camera_yaw.detach().clone()
                    previous_camera_pitch = current_camera_pitch.detach().clone()
                    intermediate_images_cpu.append(last_images)
                iteration_bar.update(1)
                iteration_bar.set_description("Loss: %5.3f" % np.sum(last_loss))
                loss_per_iteration.append(last_loss)

        loss_per_iteration = np.stack(loss_per_iteration)
        print("Done")
        print("Losses:", loss_per_iteration)
        final_images_cpu = renderer['forward-immediate'](
            camera=camera_module(current_camera_center, current_camera_yaw,
                                 current_camera_pitch, current_camera_distance),
            fov_y_radians=initial_camera_fov_radians,
            tf=tf, volume=volume_data).detach().cpu().numpy()

        return {
            'optimized_positions': optimized_positions,
            'optimized_intermediate_positions': optimized_intermediate_positions,
            'loss_per_iteration': loss_per_iteration,
            'initial_images_cpu': initial_images_cpu,
            'final_images_cpu': final_images_cpu,
            'intermediate_images_cpu': np.stack(intermediate_images_cpu),
        }

    full_results['optimization'] = optimize()

    # Save
    if save:
        print("Save")
        folder = get_result_folder("camera/entropy2d_" + name)
        np.savez_compressed(os.path.join(folder, "data.npz"),
                            **full_results)
    print("Done computing")
    return full_results

def visualize(name, show_or_export: str, data=None, hemisphere="full"):
    assert show_or_export in ["show", "export"]

    #VIS_KEY = "forward-delayed"
    #VIS_KEY = "adjoint"

    print("Load")
    folder = get_result_folder("camera/entropy2d_" + name)
    if data is None:
        data = np.load(os.path.join(folder, "data.npz"), allow_pickle=True)

    def _autolabel(rects, ax, format='%d', off_x=0, off_y=0, center=False, centerThreshold=1.0):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            if center and height>centerThreshold:
                xy = (rect.get_x() + rect.get_width() / 2, height/2)
                xytext = (off_x, off_y)
                va='center'
            else:
                xy = (rect.get_x() + rect.get_width() / 2, height)
                xytext = (off_x, off_y+3) # 3 points vertical offset
                va='bottom'
            ax.annotate(format % height,
                        xy=xy,
                        xytext=xytext,
                        textcoords="offset points",
                        ha='center', va=va,
                        fontstretch='ultra-condensed',
                        fontsize='small')

    def _item(x):
        try:
            return x.item()
        except AttributeError:
            return x

    optimization_colors = None
    def export_loss_spheres():
        nonlocal optimization_colors
        background = _item(data['background'])
        foreground = _item(data['forward-delayed'])
        background_positions = background['cameras'][:,0,:] - background['camera_center']
        background_values = background['loss_values']
        foreground_positions = foreground['cameras'][:, 0, :] - foreground['camera_center']
        foreground_values = foreground['loss_values']

        # compute next centers
        grad_stepsize = 2
        camera_module = CameraOnASphere(foreground['camera_orientation'])
        camera_yaw = torch.from_numpy(foreground['camera_yaw']+grad_stepsize*foreground['grad_yaw']).unsqueeze(1)
        camera_pitch = torch.from_numpy(foreground['camera_pitch']+grad_stepsize*foreground['grad_pitch']).unsqueeze(1)
        camera_distance = torch.from_numpy(foreground['camera_distance']).unsqueeze(1)
        camera_center = torch.from_numpy(foreground['camera_center'])
        next_positions = camera_module(
            camera_center, camera_yaw, camera_pitch, camera_distance)[:, 0, :] - foreground['camera_center']

        optimization = _item(data['optimization'])
        optimized_positions = [c[:,0,:] for c in optimization['optimized_positions']] - foreground['camera_center'][0,:][np.newaxis,:]
        optimized_intermediate_positions = optimization['optimized_intermediate_positions'] - foreground['camera_center'][0,:][np.newaxis,:]
        M = optimized_intermediate_positions[0].shape[0]

        all_positions = np.concatenate([background_positions, foreground_positions])
        all_values = np.concatenate([background_values, foreground_values])
        min_cost = np.min(all_values)
        max_cost = np.max(all_values)
        bounds = np.max(np.abs(all_positions))
        norm = matplotlib.colors.Normalize(vmin=min_cost, vmax=max_cost)

        original_cmap = matplotlib.cm.get_cmap('RdBu_r')

        def cmap_map(function, cmap):
            """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
            This routine will break any discontinuous points in a colormap.
            """
            cdict = cmap._segmentdata
            step_dict = {}
            # Firt get the list of points where the segments start or end
            for key in ('red', 'green', 'blue'):
                step_dict[key] = list(map(lambda x: x[0], cdict[key]))
            step_list = sum(step_dict.values(), [])
            step_list = np.array(list(set(step_list)))
            # Then compute the LUT, and apply the function to the LUT
            reduced_cmap = lambda step: np.array(cmap(step)[0:3])
            old_LUT = np.array(list(map(reduced_cmap, step_list)))
            new_LUT = np.array(list(map(function, old_LUT)))
            # Now try to make a minimal segment definition of the new LUT
            cdict = {}
            for i, key in enumerate(['red', 'green', 'blue']):
                this_cdict = {}
                for j, step in enumerate(step_list):
                    if step in step_dict[key]:
                        this_cdict[step] = new_LUT[j, i]
                    elif new_LUT[j, i] != old_LUT[j, i]:
                        this_cdict[step] = new_LUT[j, i]
                colorvector = list(map(lambda x: x + (x[1],), this_cdict.items()))
                colorvector.sort()
                cdict[key] = colorvector

            return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)
        if hemisphere=='full':
            light_cmap = original_cmap
        else:
            light_cmap = cmap_map(lambda x: x/2 + 0.5, original_cmap)

        fig, axs = plt.subplots(1, 2, figsize=(6, 2.5))
        axs[0].set_title("North")
        axs[1].set_title("South")
        for i, ax in enumerate(axs):
            # filter by z
            if i == 0:
                mask = all_positions[:, 2] >= 0
                mask2 = foreground_positions[:, 2] >= 0
            else:
                mask = all_positions[:, 2] < 0
                mask2 = foreground_positions[:, 2] < 0
            x = all_positions[mask, 0]
            y = all_positions[mask, 1]
            z = all_values[mask]
            im = ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k', norm=norm)
            ax.tricontourf(x, y, z, levels=14, cmap=light_cmap, norm=norm)
            if hemisphere is not 'small':
                ax.plot(foreground_positions[mask2, 0], foreground_positions[mask2, 1], 'ko', ms=2)
            ax.set_aspect('equal')
            ax.set_xlim(-bounds, +bounds)
            ax.set_ylim(-bounds, +bounds)

            #gradients
            if hemisphere is not 'small':
                x1 = foreground_positions[mask2, 0]
                y1 = foreground_positions[mask2, 1]
                x2 = next_positions[mask2, 0]
                y2 = next_positions[mask2, 1]
                for (x_start, y_start, x_end, y_end) in zip(x1, y1, x2, y2):
                    arrow_len = np.sqrt((x_start-x_end)**2+(y_start-y_end)**2)
                    ax.annotate('', xytext=(x_start, y_start), xy=(x_end, y_end),
                                arrowprops=dict(arrowstyle="->", color='k', alpha=0.5),
                                size=int(100*arrow_len))

            # optimization paths
            optimization_colors = []
            for m in range(M):
                local_positions1 = np.array([p[m, :] for p in optimized_intermediate_positions])
                local_positions2 = np.array([p[m, :] for p in optimized_positions])
                if i == 0:
                    mask1 = local_positions1[:, 2] >= 0
                    mask2 = local_positions2[:, 2] >= 0
                else:
                    mask1 = local_positions1[:, 2] < 0
                    mask2 = local_positions2[:, 2] < 0
                x1 = local_positions1[mask1, 0]
                y1 = local_positions1[mask1, 1]
                x2 = local_positions2[mask2, 0]
                y2 = local_positions2[mask2, 1]
                line = ax.plot(x1, y1, ms=1)[0]
                optimization_colors.append(line.get_color())
                last_pos = None
                for j in range(1, len(x2), 10):
                    pos_end = (x2[j], y2[j])
                    pos_start = (0.1 * x2[j - 1] + 0.9 * x2[j], 0.1 * y2[j - 1] + 0.9 * y2[j])
                    if last_pos is not None:
                        dist = np.sqrt((last_pos[0]-x2[j])**2 + (last_pos[1]-y2[j])**2)
                        if dist < 0.15: continue
                    last_pos = pos_end
                    ax.annotate('', xytext=pos_start, xy=pos_end,
                                arrowprops=dict(arrowstyle="->", color=line.get_color()),
                                size=10)

        # colorbar
        cbar_ax = fig.add_axes([0.9, 0.05, 0.03, 0.9])
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=light_cmap), cax=cbar_ax)
        #fig.colorbar(im, cax=cbar_ax)

        plt.subplots_adjust(0.06, 0.10, 0.88, 0.90, 0.15, 0.20)
        if show_or_export == "show":
            plt.show()
        else:
            plt.savefig(os.path.join(folder, "losses.pdf"))
            plt.close(fig)
    export_loss_spheres()

    def export_timings():
        keys = ['forward-immediate', 'forward-delayed', 'adjoint']
        forward_times_ms = dict([(key, _item(data[key])['forward_times_ms']) for key in keys])
        backward_times_ms = dict([(key, _item(data[key])['backward_times_ms']) for key in keys])
        forward_memory_bytes = dict([(key, _item(data[key])['forward_memory_bytes']) for key in keys])

        values_names = ["Forward pass", "Backward pass", "Memory"]
        X = np.arange(len(values_names))
        offsets = np.linspace(-0.3, +0.3, len(keys))
        width = (offsets[1] - offsets[0]) * 0.9

        fig = plt.figure(figsize=(5, 5/2))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        for i, key in enumerate(keys):
            ft = forward_times_ms[key][1:]
            bt = backward_times_ms[key][1:]
            fm = forward_memory_bytes[key] / (1024 * 1024)
            ft_mean, ft_std = np.mean(ft), np.std(ft)
            bt_mean, bt_std = np.mean(bt), np.std(bt)
            fm_mean, fm_std = np.mean(fm), np.std(fm)
            rects0 = ax0.bar(X[:2] + offsets[i], [ft_mean, bt_mean], width,
                             yerr=[ft_std, bt_std], align='center', capsize=10, label=key)
            #_autolabel(rects0, ax0, off_x=-9)
            _autolabel(rects0, ax0, format="%.2f", off_y=0, center=True)
            rects1 = ax1.bar(X[2:] + offsets[i], [fm_mean], width,
                             yerr=[fm_std], align='center', capsize=10, label=key)
            _autolabel(rects1, ax1, format="%.1f", off_y=-2, center=True)

        ax0.set_ylabel("Time (ms)")
        ax0.set_xticks(X[:2])
        ax0.set_xticklabels(values_names[:2])
        ax1.set_ylabel("MB")
        ax1.set_xticks(X[2:])
        ax1.set_xticklabels(values_names[2:])

        # ax.legend()
        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3)

        plt.subplots_adjust(
            top=0.855,
            bottom=0.105,
            left=0.098,
            right=0.975,
            hspace=0.2,
            wspace=0.365)
        if show_or_export == "show":
            plt.show()
        else:
            plt.savefig(os.path.join(folder, "timings.pdf"))
            plt.close(fig)
    export_timings()

    best_optim_loss = 0
    def save_img(img, prefix):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(folder, prefix + '.png'), img)

    def export_optim_losses():
        nonlocal best_optim_loss
        optimization = _item(data['optimization'])
        loss_values = optimization['loss_per_iteration']
        initial_images = optimization['initial_images_cpu']
        final_images = optimization['final_images_cpu']
        intermediate_images = optimization['intermediate_images_cpu']

        foreground = _item(data['forward-delayed'])
        foreground_values = foreground['loss_values']
        best_index = np.argmax(foreground_values)
        best_sample_loss = foreground_values[best_index]

        X = np.arange(len(loss_values)-1)
        M = loss_values[0].shape[0]

        fig = plt.figure(figsize=(6,5/2))
        for m in range(M):
            Y = [loss_values[x][m] for x in range(1,len(loss_values))]
            #plt.plot(X, Y)
            plt.semilogy(X, Y)
        plt.gca().axhline(best_sample_loss, color='k', linestyle='--')
        plt.gca().set_xticks([0,5,10,15,20])
        plt.gca().yaxis.set_minor_formatter(mticker.ScalarFormatter())

        best_run = np.argmax(loss_values[-1])
        print("Best run index:", best_run)
        fig.suptitle("Best run, loss: %.3f"%(loss_values[-1][best_run]))
        best_optim_loss = loss_values[-1][best_run]

        plt.subplots_adjust(
            top=0.91,
            bottom=0.095,
            left=0.07,
            right=0.975,
            hspace=0.2,
            wspace=0.305)
        if show_or_export == "show":
            plt.show()
        else:
            plt.savefig(os.path.join(folder, "optim.pdf"))
            plt.close(fig)
            save_img(initial_images[best_run, ...], 'best-initial')
            save_img(final_images[best_run, ...], 'best-final')
            for i in range(intermediate_images.shape[0]):
                save_img(intermediate_images[i, best_run, ...], 'best-%d'%i)

    export_optim_losses()


    def export_optim_losses_small():
        optimization = _item(data['optimization'])
        loss_values = optimization['loss_per_iteration']
        M = loss_values[0].shape[0]
        loss_values = np.array([loss_values[-1][m] for m in range(M)])
        colors = optimization_colors

        background = _item(data['background'])
        foreground = _item(data['forward-delayed'])
        background_values = background['loss_values']
        foreground_values = foreground['loss_values']
        #all_values = np.concatenate([background_values, foreground_values])

        fig = plt.figure(figsize=(3, 5 / 2))
        ax = plt.gca()

        # jitter plot for optimizations
        def rand_jitter(arr):
            stdev = 0.05  # .1 * (max(arr) - min(arr))
            return arr + np.random.randn(len(arr)) * stdev

        def jitter(ax, x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None,
                   linewidths=None, verts=None, hold=None, **kwargs):
            return ax.scatter(rand_jitter(x), y, s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                              alpha=alpha, linewidths=linewidths, **kwargs)

        jitter(ax, np.ones_like(loss_values), loss_values, c=colors)

        # violin plot for sampling
        ax.violinplot([foreground_values], positions=[2])

        # annotations
        r = np.max(foreground_values)-np.min(foreground_values)
        ax.annotate("%.5f" % np.max(loss_values), (1.0, np.max(loss_values) + .03*r), ha='center')
        ax.annotate("%.5f" % np.max(foreground_values), (2.0, np.max(foreground_values) + .03*r), ha='center')

        ax.set_xticks([1, 2])
        ax.set_xlim(1 - 0.6, 2 + 0.6)
        ax.set_ylim(top=max(np.max(loss_values), np.max(foreground_values)) + .1*r)
        ax.set_xticklabels(["Optimization", "Sampling"])
        #ax.set_ylabel("Loss Value")
        #fig.suptitle("Loss Value")

        plt.subplots_adjust(
            top=0.98, #0.91,
            bottom=0.11,
            left=0.17,
            right=0.96,
            hspace=0.2,
            wspace=0.305)
        if show_or_export == "show":
            plt.show()
        else:
            plt.savefig(os.path.join(folder, "optim-small.pdf"))
            plt.close(fig)

    export_optim_losses_small()


    def export_sampled():
        foreground = _item(data['forward-delayed'])
        foreground_values = foreground['loss_values']
        foreground_images = foreground['images']
        best_index = np.argmax(foreground_values)
        worst_index = np.argmin(foreground_values)
        best_sample_loss = foreground_values[best_index]
        worst_sample_loss = foreground_values[worst_index]
        if show_or_export=='export':
            save_img(foreground_images[best_index, ...], 'best-sampled')
            save_img(foreground_images[worst_index, ...], 'worst-sampled')
            with open(os.path.join(folder, "optim.txt"), 'w') as f:
                f.write("best sampled entropy: %f\n"%best_sample_loss)
                f.write("best optimized entropy: %f\n"%best_optim_loss)
                f.write("worst sampled entropy: %f\n" % worst_sample_loss)

    export_sampled()

def run(settings_file, name, show_or_export, include_color_in_entropy, params=None, hemisphere="full"):
    data = compute(settings_file, name, include_color_in_entropy, save=False, params=params)
    visualize(name, show_or_export, data, hemisphere=hemisphere)


if __name__ == '__main__':
    run("../../config-files/tooth1.json", "Tooth-WColor", "export", True, hemisphere="light")
    run("../../config-files/single_jet.json", "Jet", "export", False, {'lr':10, 'iterations':20, 'intermediate':10}, hemisphere="small")
    run("../../config-files/c60multi-pw.json", "C60multi", "export", False, hemisphere="small")
    run("../../config-files/plume100-linear.json", "plume100", "export", False, {'lr':10.0, 'iterations':20, 'intermediate':200}, hemisphere="small")
