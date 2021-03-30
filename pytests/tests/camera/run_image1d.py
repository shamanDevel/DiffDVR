import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tqdm
import imageio

from diffdvr import Renderer, CameraOnASphere, Entropy, ColorMatches, Settings, setup_default_settings, \
    fibonacci_sphere, renderer_dtype_torch, renderer_dtype_np, ProfileRenderer
from results import get_result_folder
from tests import CommonVariables
import pyrenderer


def compute(settings_file, name):
    N = 360
    Batches = 20
    assert N % Batches == 0

    # settings
    s = Settings(settings_file)
    volume = s.load_dataset()
    volume.copy_to_gpu()
    volume_data = volume.getDataGpu(0) if CommonVariables.run_on_cuda else volume.getDataCpu(0)
    device = volume_data.device

    rs = setup_default_settings(
        volume, CommonVariables.screen_width, CommonVariables.screen_height,
        s.get_stepsize(), CommonVariables.run_on_cuda)
    rs.tf_mode = pyrenderer.TFMode.Linear
    tf_points = s.get_tf_points()
    tf = pyrenderer.TFUtils.get_piecewise_tensor(tf_points)
    tf = tf.to(dtype=renderer_dtype_torch, device=device)

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
    default_renderer = renderer['forward-delayed']

    # camera
    camera_config = s.get_camera()
    camera_yaw_cpu = np.linspace(0, 2 * np.pi, N, endpoint=False, dtype=renderer_dtype_np)
    camera_pitch_cpu = camera_config.pitch_radians * np.ones((N,), dtype=renderer_dtype_np)
    camera_distance_cpu = camera_config.distance * np.ones((N,), dtype=renderer_dtype_np)
    camera_center_cpu = np.stack([camera_config.center] * N, axis=0).astype(dtype=renderer_dtype_np)
    camera_fov_radians = camera_config.fov_y_radians
    camera_module = CameraOnASphere(camera_config.orientation)

    # allocate outputs
    loss_values_cpu = np.zeros((N,), dtype=renderer_dtype_np)
    grad_yaw_cpu = np.zeros((N,), dtype=renderer_dtype_np)
    cameras_cpu = np.zeros((N, 3, 3), dtype=renderer_dtype_np)
    images_cpu = np.zeros((N, CommonVariables.screen_height, CommonVariables.screen_width, 4), dtype=renderer_dtype_np)
    renderer_forward_milliseconds = dict()
    renderer_backward_milliseconds = dict()
    renderer_forward_memory = dict()

    # reference
    print("Render reference image")

    def render_reference():
        camera_yaw = torch.tensor([camera_config.yaw_radians],
                                  dtype=renderer_dtype_torch, device=device).unsqueeze(1)
        camera_pitch = torch.tensor([camera_config.pitch_radians],
                                    dtype=renderer_dtype_torch, device=device).unsqueeze(1)
        camera_distance = torch.tensor([camera_config.distance],
                                       dtype=renderer_dtype_torch, device=device).unsqueeze(1)
        camera_center = torch.tensor([list(camera_config.center)],
                                     dtype=renderer_dtype_torch, device=device)
        viewport = camera_module(camera_center, camera_yaw, camera_pitch, camera_distance)
        images = default_renderer(camera=viewport, fov_y_radians=camera_fov_radians,
                                  tf=tf, volume=volume_data)
        return images.detach()

    reference_image = render_reference()
    reference_image_cpu = reference_image.cpu().numpy()

    # create loss
    class ImageLoss(torch.nn.Module):
        def __init__(self, reference):
            super().__init__()
            self.register_buffer("_reference", reference)
            self._loss = torch.nn.MSELoss(reduction='none')

        def forward(self, x):
            return torch.mean(self._loss(x, self._reference), dim=(1, 2, 3))

    loss = ImageLoss(reference_image)
    loss.to(device=device, dtype=renderer_dtype_torch)

    # render
    for renderer_name, renderer_instance in renderer.items():
        print("Render", renderer_name)
        forward_ms = np.zeros(N // Batches)
        forward_memory = np.zeros(N // Batches)
        backward_ms = np.zeros(N // Batches)
        for b in tqdm.trange(N // Batches):
            istart = b * Batches;
            iend = istart + Batches
            camera_yaw = torch.from_numpy(camera_yaw_cpu[istart:iend]).to(device=device).unsqueeze(1)
            camera_pitch = torch.from_numpy(camera_pitch_cpu[istart:iend]).to(device=device).unsqueeze(1)
            camera_distance = torch.from_numpy(camera_distance_cpu[istart:iend]).to(device=device).unsqueeze(1)
            camera_center = torch.from_numpy(camera_center_cpu[istart:iend]).to(device=device)
            camera_yaw.requires_grad_(True)
            cameras = camera_module(camera_center, camera_yaw, camera_pitch, camera_distance)
            cameras_cpu[istart:iend, ...] = cameras.detach().cpu().numpy()

            profiling = ProfileRenderer()
            images = renderer_instance(
                camera=cameras, fov_y_radians=camera_fov_radians,
                tf=tf, volume=volume_data, profiling=profiling)

            loss_values = loss(images)

            loss_scalar = torch.sum(loss_values)
            loss_scalar.backward()

            forward_ms[b] = profiling.forward_ms / Batches
            forward_memory[b] = profiling.forward_bytes / Batches
            backward_ms[b] = profiling.backward_ms / Batches

            if renderer_instance is default_renderer:
                loss_values_cpu[istart:iend, ...] = loss_values.detach().cpu().numpy()
                images_cpu[istart:iend, ...] = images.detach().cpu().numpy()
                grad_yaw_cpu[istart:iend, ...] = camera_yaw.grad[:, 0].detach().cpu().numpy()
        renderer_forward_milliseconds[renderer_name] = forward_ms
        renderer_backward_milliseconds[renderer_name] = backward_ms
        renderer_forward_memory[renderer_name] = forward_memory

    print("forward times:", renderer_forward_milliseconds)
    print("backward times:", renderer_backward_milliseconds)
    print("forward memory:", renderer_forward_memory)

    # Save
    print("Save")
    folder = get_result_folder("camera/image1d_" + name)
    np.savez_compressed(os.path.join(folder, "data.npz"),
                        camera_yaws=camera_yaw_cpu,
                        cameras=cameras_cpu,
                        reference_yaw=camera_config.yaw_radians,
                        reference_image=reference_image_cpu,
                        images=images_cpu,
                        loss_values=loss_values_cpu,
                        grad_yaws=grad_yaw_cpu,
                        forward_times_ms=renderer_forward_milliseconds,
                        backward_times_ms=renderer_backward_milliseconds,
                        forward_memory_bytes=renderer_forward_memory)

    print("Done computing")


def visualize(name, show_or_export: str):
    assert show_or_export in ["show", "export"]

    print("Load")
    folder = get_result_folder("camera/image1d_" + name)
    data = np.load(os.path.join(folder, "data.npz"), allow_pickle=True)

    def _autolabel(rects, ax, format='%d', off_x=0):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(format % height,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(off_x, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    def export_timings():
        forward_times_ms = data['forward_times_ms'].item()
        backward_times_ms = data['backward_times_ms'].item()
        forward_memory_bytes = data['forward_memory_bytes'].item()
        keys = forward_times_ms.keys()

        values_names = ["Forward", "Backward", "Memory"]
        X = np.arange(len(values_names))
        offsets = np.linspace(-0.3, +0.3, len(keys))
        width = (offsets[1] - offsets[0]) * 0.9

        fig = plt.figure(figsize=(6, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        for i, key in enumerate(keys):
            ft = forward_times_ms[key]
            bt = backward_times_ms[key]
            fm = forward_memory_bytes[key] / (1024 * 1024)
            ft_mean, ft_std = np.mean(ft), np.std(ft)
            bt_mean, bt_std = np.mean(bt), np.std(bt)
            fm_mean, fm_std = np.mean(fm), np.std(fm)
            rects0 = ax0.bar(X[:2] - offsets[i], [ft_mean, bt_mean], width,
                             yerr=[ft_std, bt_std], align='center', capsize=10, label=key)
            _autolabel(rects0, ax0, off_x=-9)
            rects1 = ax1.bar(X[2:] - offsets[i], [fm_mean], width,
                             yerr=[fm_std], align='center', capsize=10, label=key)
            _autolabel(rects1, ax1)

        ax0.set_ylabel("Time (ms)")
        ax0.set_xticks(X[:2])
        ax0.set_xticklabels(values_names[:2])
        ax1.set_ylabel("mBytes")
        ax1.set_xticks(X[2:])
        ax1.set_xticklabels(values_names[2:])

        # ax.legend()
        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3)

        plt.subplots_adjust(0.088, 0.07, 0.979, 0.92, 0.305, 0.2)
        if show_or_export == "show":
            plt.show()
        else:
            plt.savefig(os.path.join(folder, "timings.pdf"))
            plt.close(fig)

    export_timings()

    def find_optima():
        minima = []
        maxima = []
        loss_values = data['loss_values']
        N = loss_values.shape[0]
        for i in range(N):
            current = loss_values[i]
            previous = loss_values[i - 1] if i > 0 else loss_values[-1]
            next = loss_values[i + 1] if i < N - 1 else loss_values[0]
            if (current > previous) and (current > next):
                maxima.append(i)
            if (current < previous) and (current < next):
                minima.append(i)
        return np.array(minima, dtype=np.int32), np.array(maxima, dtype=np.int32)

    minima, maxima = find_optima()
    all_optima = list(sorted(list(minima) + list(maxima)))
    labels = [chr(ord('a') + i) for i in range(len(all_optima))]
    minima_labels = [labels[all_optima.index(i)] for i in minima]
    maxima_labels = [labels[all_optima.index(i)] for i in maxima]

    def export_losses():
        yaw_values = np.rad2deg(data['camera_yaws'])
        loss_values = data['loss_values']
        yaw_grad = data['grad_yaws']

        fig = plt.figure(figsize=(7, 5))

        color = 'tab:blue'
        axE = plt.gca()
        axE.set_xlabel("Longitude (degrees)")
        axE.set_ylabel("Loss", color=color)
        axE.plot(yaw_values, loss_values, color=color)
        axE.tick_params(axis='y', labelcolor=color)

        axE2 = axE.twinx()
        color = 'tab:red'
        axE2.set_ylabel("Gradient", color=color)
        axE2.plot(yaw_values, yaw_grad, color=color)
        axE2.tick_params(axis='y', labelcolor=color)
        axE2.axhline(0.0, color=color, linestyle="--", alpha=0.5)

        # reference
        reference_yaw = np.mod(data['reference_yaw'], 2 * np.pi)
        closest_index = np.argmin((data['camera_yaws'] - reference_yaw) ** 2)
        print(closest_index, data['reference_yaw'], yaw_values[closest_index])
        line = axE.plot([yaw_values[closest_index]], [loss_values[closest_index]], 'ko', ms=5)
        axE.annotate('gt',
                     xy=(yaw_values[closest_index], loss_values[closest_index]),
                     xytext=(-5, 0),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='right', va='center')

        # minima and maxima
        axE.plot(yaw_values[minima], loss_values[minima], 'ko', ms=2)
        for idx, label in zip(minima, minima_labels):
            axE.annotate(label + ')',
                         xy=(yaw_values[idx], loss_values[idx]),
                         xytext=(0, -3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='top')
        axE.plot(yaw_values[maxima], loss_values[maxima], 'ko', ms=2)
        for idx, label in zip(maxima, maxima_labels):
            axE.annotate(label + ')',
                         xy=(yaw_values[idx], loss_values[idx]),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

        fig.suptitle("Loss value per angle")
        plt.subplots_adjust(0.1, 0.1, 0.883, 0.921)
        if show_or_export == "show":
            plt.show()
        else:
            plt.savefig(os.path.join(folder, "loss.pdf"))
            plt.close(fig)

    export_losses()

    def export_images():
        images = data['images']
        for idx, label in zip(all_optima, labels):
            img = images[idx, ...] * 255
            img = np.clip(img, 0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(folder, 'image-' + label + '.png'), img)

    if show_or_export == "export":
        export_images()


def run(settings_file, name, show_or_export):
    compute(settings_file, name)
    visualize(name, show_or_export)


if __name__ == '__main__':
    run("../../config-files/tooth1.json", "Tooth", "export")
    # run("../../config-files/single_jet.json", "Jet", "export")
