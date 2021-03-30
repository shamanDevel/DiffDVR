
import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import tqdm
import imageio

from diffdvr import Renderer, CameraOnASphere, Settings, setup_default_settings, \
    fibonacci_sphere, renderer_dtype_torch, renderer_dtype_np, ProfileRenderer, \
    TfTexture, SmoothnessPrior, toCHW, toHWC
from results import get_result_folder
from tests import CommonVariables
import losses.ssim
from losses.lossbuilder import LossBuilder
import pyrenderer


def compute(settings_file, name, smoothness_prior,
            iterations = 50, optimizer = torch.optim.Adam,
            all_renderer = False, Min_R = 6, Max_R=6):
    NumViews = 8
    Max_R_for_forward = 4
    Seed = 124
    torch.manual_seed(Seed)
    np.random.seed(Seed)
    torch.set_num_threads(4)

    optimizer_class = optimizer
    optimizer_params = {'lr': 0.8}
    optimizer_num_iterations = 4
    optimizer_num_iterations_adjoint_delayed = iterations

    # settings
    s = Settings(settings_file)
    volume = s.load_dataset()
    volume.copy_to_gpu()
    volume_data = volume.getDataGpu(0) if CommonVariables.run_on_cuda else volume.getDataCpu(0)
    device = volume_data.device

    # camera
    camera_config = s.get_camera()
    camera_pitch_cpu, camera_yaw_cpu = fibonacci_sphere(NumViews, dtype=renderer_dtype_np)
    camera_distance_cpu = camera_config.distance * np.ones((NumViews,), dtype=renderer_dtype_np)
    camera_center_cpu = np.stack([camera_config.center] * NumViews, axis=0).astype(dtype=renderer_dtype_np)
    camera_fov_radians = camera_config.fov_y_radians
    camera_module = CameraOnASphere(camera_config.orientation)
    cameras = camera_module(
        torch.from_numpy(camera_center_cpu).to(device=device),
        torch.from_numpy(camera_yaw_cpu).to(device=device).unsqueeze(1),
        torch.from_numpy(camera_pitch_cpu).to(device=device).unsqueeze(1),
        torch.from_numpy(camera_distance_cpu).to(device=device).unsqueeze(1))

    # tf
    rs = setup_default_settings(
        volume, CommonVariables.screen_width, CommonVariables.screen_height,
        s.get_stepsize(), CommonVariables.run_on_cuda)
    rs.tf_mode = pyrenderer.TFMode.Texture
    tf_points = s.get_tf_points()
    tf_module = TfTexture()
    def random_tf(R: int):
        return torch.randn((1, R, 4), dtype=renderer_dtype_torch, device=device)

    # renderer
    renderer = {}
    if all_renderer:
        renderer['forward-immediate'] = Renderer(rs, optimize_tf=True,
                                      gradient_method='forward',
                                      forward_delayed_gradients=False)
        renderer['forward-delayed'] = Renderer(rs, optimize_tf=True,
                                                 gradient_method='forward',
                                                 forward_delayed_gradients=True)
        renderer['adjoint-immediate'] = Renderer(rs, optimize_tf=True,
                                      gradient_method='adjoint',
                                      tf_delayed_accumulation=False)
    renderer['adjoint-delayed'] = Renderer(rs, optimize_tf=True,
                                           gradient_method='adjoint',
                                           tf_delayed_accumulation=True)
    default_renderer = renderer['adjoint-delayed']

    def render_images(renderer_instance, tf_parameterized, profiling=None):
        tf = tf_module(tf_parameterized)
        images = renderer_instance(
            camera=cameras, fov_y_radians=camera_fov_radians,
            tf=tf, volume=volume_data, profiling=profiling)
        return images, tf

    full_results = {}

    # REFERENCE
    print("Render reference with R=256")
    def renderReference():
        tf_parameterized = TfTexture.init_from_points(tf_points, 256)
        images, tf = render_images(
            default_renderer,
            tf_parameterized.to(dtype=renderer_dtype_torch, device=device))
        full_results['reference'] = {
            'tf': tf.detach().cpu().numpy(),
            'images': images.detach().cpu().numpy()
        }
        return images

    reference_images = renderReference()

    # Define loss
    class ImageLoss(torch.nn.Module):
        def __init__(self, reference):
            super().__init__()
            self.register_buffer("_reference", reference)
            #self._loss = torch.nn.MSELoss()
            self._loss = torch.nn.L1Loss()
            self._prior_weight = smoothness_prior
            self._prior = SmoothnessPrior(1)

        def forward(self, x, tf):
            content_loss = self._loss(x, self._reference)
            prior_loss = self._prior(tf[:,:,:3])
            #print("content loss:", content_loss.item(), ", prior loss:", prior_loss.item(), "(", self._prior_weight, "x)")
            return content_loss + self._prior_weight * prior_loss

    loss = ImageLoss(reference_images)
    loss.to(device=device, dtype=renderer_dtype_torch)

    # Optimize
    rx = [2**i for i in range(Min_R, Max_R+1)]
    full_results['resolutions'] = np.array(rx)
    for r in rx:
        print("Optimize for resolutions", r)
        def optimize(r):
            local_results = {}
            initial_tf_parameterized = random_tf(r)

            # initial images for visualization
            initial_images, initial_tf = render_images(
                default_renderer, initial_tf_parameterized)
            local_results['initial_images'] = initial_images.detach().cpu().numpy()
            local_results['initial_tf'] = initial_tf.detach().cpu().numpy()
            del initial_images, initial_tf

            for renderer_key, renderer_instance in renderer.items():
                if r > 2**Max_R_for_forward and ('forward' in renderer_key):
                    print("Skip", renderer_key)
                    continue
                print("Optimize with", renderer_key)
                # init optimizer
                forward_ms, forward_bytes, backward_ms = [], [], []
                current_tf_parameterized = initial_tf_parameterized.clone()
                current_tf_parameterized.requires_grad_(True)
                optimizer = optimizer_class([current_tf_parameterized], **optimizer_params)
                last_loss = None
                last_images = None
                last_tf = None
                def optim_closure():
                    nonlocal last_loss, last_images, last_tf
                    optimizer.zero_grad()
                    profiling = ProfileRenderer()
                    images, tf = render_images(
                        renderer_instance, current_tf_parameterized, profiling)
                    loss_value = loss(images, tf)
                    last_images = images
                    last_tf = tf
                    last_loss = loss_value.item()
                    loss_value.backward()
                    forward_ms.append(profiling.forward_ms)
                    backward_ms.append(profiling.backward_ms)
                    forward_bytes.append(profiling.forward_bytes)
                    return loss_value

                # allocate outputs
                optim_closure()
                optimized_tfs = [last_tf.detach().cpu().numpy()]
                optimized_losses = [last_loss]

                # optimize
                num_iterations = optimizer_num_iterations_adjoint_delayed if renderer_key == "adjoint-delayed" else optimizer_num_iterations
                with tqdm.tqdm(num_iterations + 1) as iteration_bar:
                    for iteration in range(num_iterations + 1):
                        optimizer.step(optim_closure)
                        iteration_bar.update(1)
                        iteration_bar.set_description("Loss: %7.5f" % last_loss)
                        optimized_tfs.append(last_tf.detach().cpu().numpy())
                        optimized_losses.append(last_loss)

                print("Done, losses:", optimized_losses)
                local_results[renderer_key] = {
                    'tfs': optimized_tfs,
                    'losses': optimized_losses,
                    'final_image': last_images.detach().cpu().numpy(),
                    'forward_ms': np.array(forward_ms),
                    'forward_bytes': np.array(forward_bytes),
                    'backward_ms': np.array(backward_ms)
                }

            return local_results
        full_results['r%d'%r] = optimize(r)

        # Save
        print("Save")
        folder = get_result_folder("tf/image_" + name + ("/smooth%05d"%int(smoothness_prior*10000)))
        np.savez_compressed(os.path.join(folder, "data.tmp.npz"),
                            **full_results)
        if os.path.exists(os.path.join(folder, "data.npz")):
            os.remove(os.path.join(folder, "data.npz"))
        os.rename(os.path.join(folder, "data.tmp.npz"), os.path.join(folder, "data.npz"))

    print("Done computing")


def tf_to_texture(tf : np.ndarray, out, figsize = (7,2), show_markers=False, out_format='png'):
    # process tf
    R = tf.shape[1]
    tf = tf.copy()
    rgb = tf[:1, :, :3]
    opacity = tf[0, :, 3]
    max_opacity = np.max(opacity)
    print('max opacity:', max_opacity)
    opacity /= max_opacity
    X = [0, 0] + list(np.arange(R) / R + 1 / (2 * R)) + [1, 1, 0]
    Y = [0, opacity[0]] + list(opacity) + [opacity[-1], 0, 0]

    # plot
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    line, = ax.plot(X[:-2], Y[:-2], 'ko-' if show_markers else 'k-')
    clip_path = Polygon(np.stack([X, Y], axis=1), edgecolor='none', closed=True)
    im = ax.imshow(rgb, aspect='auto', extent=[0, 1, 0, 1], origin='lower', zorder=line.get_zorder() - 1)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    plt.subplots_adjust(0, 0, 1, 1)
    for s in ['top', 'right', 'bottom', 'left']:
        ax.spines[s].set_visible(False)
    plt.savefig(out, format=out_format, transparent=True)
    plt.close(fig)

def load_settings(settings_file):
    print("Load settings", settings_file)
    s = Settings(settings_file)
    # volume
    volume = s.load_dataset()
    volume.copy_to_gpu()
    volume_data = volume.getDataGpu(0) if CommonVariables.run_on_cuda else volume.getDataCpu(0)
    device = volume_data.device
    # camera
    camera_config = s.get_camera()
    camera_yaw = camera_config.yaw_radians * torch.ones((1, 1), dtype=renderer_dtype_torch)
    camera_pitch = camera_config.pitch_radians * torch.ones((1, 1), dtype=renderer_dtype_torch)
    camera_distance = camera_config.distance * torch.ones((1, 1), dtype=renderer_dtype_torch)
    camera_center = torch.from_numpy(np.array([camera_config.center])).to(dtype=renderer_dtype_torch)
    camera_fov_radians = camera_config.fov_y_radians
    camera_module = CameraOnASphere(camera_config.orientation)
    cameras = camera_module(camera_center, camera_yaw, camera_pitch, camera_distance)
    cameras = cameras.to(device=device)
    # renderer
    rs = setup_default_settings(
        volume, CommonVariables.screen_width, CommonVariables.screen_height,
        s.get_stepsize(), CommonVariables.run_on_cuda)
    rs.tf_mode = pyrenderer.TFMode.Texture
    renderer = Renderer(rs)

    def render_with_tf(tf):
        tf = torch.from_numpy(tf).to(device=device, dtype=renderer_dtype_torch)
        return renderer(camera=cameras, fov_y_radians=camera_fov_radians,
                        tf=tf, volume=volume_data).detach().cpu().numpy()[0]

    return s, volume, device, cameras, renderer, render_with_tf

# Compute SSIM + PSNR
def blend_to_white(image_np):
    img = torch.from_numpy(image_np)
    assert img.shape[2] == 4
    color = img[:,:,:3]
    alpha = img[:,:,3:]
    white = torch.ones_like(color)
    return alpha * color + (1-alpha) * white

class SSIM():
    def __init__(self):
        self._ssim = LossBuilder(torch.device("cpu")).ssim_loss(3)
    def __call__(self, x, y):
        return self._ssim(toCHW(x.unsqueeze(0)), toCHW(y.unsqueeze(0))).item()
ssim = SSIM()
class LPIPS():
    def __init__(self):
        self._lpips = LossBuilder(torch.device("cpu")).lpips_loss(3, 0.0, 1.0)
    def __call__(self, x, y):
        return self._lpips(toCHW(x.unsqueeze(0)), toCHW(y.unsqueeze(0))).item()
lpips = LPIPS()
class PSNR():
    def __call__(self, x, y):
        return 10 * torch.log10(1 / torch.nn.functional.mse_loss(
            x, y, reduction='mean')).item()
psnr = PSNR()

def visualize_smoothingPriors(settings_file, name):
    # collect smoothing priors
    main_folder = get_result_folder("tf/image_" + name)
    output_folder = get_result_folder("tf/image_" + name + "/comparison")
    smoothness_priors = [o[6:] for o in os.listdir(main_folder)
                         if os.path.isdir(os.path.join(main_folder, o)) and o.startswith("smooth")]
    smoothness_priors.sort()
    print("Smoothing priors:", smoothness_priors)

    # convert to float
    X = np.array([int(p)/10000.0 for p in smoothness_priors])
    print("X:", X)

    s, volume, device, cameras, renderer, render_with_tf = \
        load_settings(settings_file)

    def save_img(img, prefix):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(output_folder, prefix + '.png'), img)

    # For every prior, render the images
    images = [None] * len(smoothness_priors)
    tfs = [None] * len(smoothness_priors)
    reference_tf = None
    reference_image = None
    resolution_to_use = None
    for i,pstr in enumerate(smoothness_priors):
        # load
        folder = get_result_folder("tf/image_" + name + "/smooth" + pstr)
        data = np.load(os.path.join(folder, "data.npz"), allow_pickle=True)
        if resolution_to_use is None:
            resolutions = []
            for r in data['resolutions']:
                if ('r%d' % r) in data:
                    resolutions.append(r)
            resolutions.sort()
            resolution_to_use = resolutions[-1]
            print("Resolutions found:", resolutions, "use", resolution_to_use)

        # fetch TF
        renderer_key = 'adjoint-delayed'
        local_result = data['r%d'%resolution_to_use].item()
        local_tf = local_result[renderer_key]['tfs'][-1]
        local_image = render_with_tf(local_tf)
        tfs[i] = local_tf
        images[i] = local_image
        tf_to_texture(local_tf, os.path.join(output_folder, "tf-prior%s-final.png" % pstr))
        save_img(local_image, "image-prior%s-final" % pstr)

        # reference
        if reference_tf is None:
            reference = data['reference'].item()
            reference_tf = reference['tf']
            reference_image = render_with_tf(reference_tf)
            tf_to_texture(reference_tf, os.path.join(output_folder, "reference-tf.png"))
            save_img(reference_image, "reference-img")

    Ypsnr = [None] * len(smoothness_priors)
    Yssim = [None] * len(smoothness_priors)
    Ylpips = [None] * len(smoothness_priors)
    reference_white = blend_to_white(reference_image)
    for i in range(len(smoothness_priors)):
        image = blend_to_white(images[i])
        Ypsnr[i] = psnr(image, reference_white)
        Yssim[i] = ssim(image, reference_white)
        Ylpips[i] = 1/lpips(image, reference_white)
    print("PSNR:", Ypsnr)
    print("SSIM:", Yssim)
    print("LPIPS:", Ylpips)
    best_index = np.argmax(Ylpips)
    print("Best index:", best_index, " with smoothing prior", X[best_index])

    # for the best index, save initial image as well
    folder = get_result_folder("tf/image_" + name + "/smooth" + smoothness_priors[best_index])
    data = np.load(os.path.join(folder, "data.npz"), allow_pickle=True)
    renderer_key = 'adjoint-delayed'
    local_result = data['r%d' % resolution_to_use].item()
    local_tf = local_result[renderer_key]['tfs'][0]
    local_image = render_with_tf(local_tf)
    tf_to_texture(local_tf, os.path.join(output_folder, "tf-prior%s-initial.png" % smoothness_priors[best_index]))
    save_img(local_image, "image-prior%s-initial" % smoothness_priors[best_index])

    # figure
    ticks = np.arange(len(X))
    fig, ax1 = plt.subplots(figsize=(7, 2))
    color = 'tab:red'
    ax1.set_xlabel('smoothing prior $\lambda$')
    ax1.set_ylabel('PSNR', color=color)
    ax1.plot(ticks, Ypsnr, '.-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(["%.3f" % x for x in X], rotation = 45)
    ax1.tick_params(axis='x', which='major', labelsize=7)
    for tick in ax1.get_xticklabels():
        if tick.get_text() == "%.3f"%X[best_index]:
            tick.set_fontweight('bold')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('SSIM', color=color)  # we already handled the x-label with ax1
    ax2.plot(ticks, Yssim, '.-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax3.spines['right'].set_position(("axes", 1.15))
    ax3.set_ylabel('1/LPIPS', color=color)  # we already handled the x-label with ax1
    ax3.plot(ticks, Ylpips, '.-', color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    fig.tight_layout(pad=0)  # otherwise the right y-label is slightly clipped
    plt.savefig(os.path.join(output_folder, "losses.pdf"))

    # assemble LaTeX for comparison
    LATEX_TEMPLATE = """
\\documentclass[border={-10pt 0pt 0pt 0pt}]{standalone}
\\usepackage{amsmath}
\\usepackage{graphicx}
\\usepackage{tikz}
\\usepackage{adjustbox}

\\begin{document}
	\\newlength{\\basewidth}
	\\setlength{\\basewidth}{40mm}
	\\setlength\\tabcolsep{1pt}
	\\setlength{\\fboxsep}{0pt}%
	\\begin{tabular}{cccc}
		\\begin{tikzpicture}%
			\\node[inner sep=0pt] at (0,0) {\\includegraphics[width=\\basewidth]{{reference-img.png}}};
			\\node[inner sep=0pt,fill=white] at (0,-0.4\\basewidth) {\\fbox{\\includegraphics[width=0.8\\basewidth]{{reference-tf.png}}}};
		\\end{tikzpicture}&
		\\begin{tikzpicture}%
			\\node[inner sep=0pt] at (0,0) {\\includegraphics[width=\\basewidth]{{image-prior00000-final.png}}};
			\\node[inner sep=0pt,fill=white] at (0,-0.4\\basewidth) {\\fbox{\\includegraphics[width=0.8\\basewidth]{{tf-prior#MinStr#-final.png}}}};
		\\end{tikzpicture}&
		\\begin{tikzpicture}%
			\\node[inner sep=0pt] at (0,0) {\\includegraphics[width=\\basewidth]{{image-prior02000-final.png}}};
			\\node[inner sep=0pt,fill=white] at (0,-0.4\\basewidth) {\\fbox{\\includegraphics[width=0.8\\basewidth]{{tf-prior#BestStr#-final.png}}}};
		\\end{tikzpicture}&
		\\begin{tikzpicture}%
			\\node[inner sep=0pt] at (0,0) {\\includegraphics[width=\\basewidth]{{image-prior05000-final.png}}};
			\\node[inner sep=0pt,fill=white] at (0,-0.4\\basewidth) {\\fbox{\\includegraphics[width=0.8\\basewidth]{{tf-prior#MaxStr#-final.png}}}};
		\\end{tikzpicture}\\\\
		reference & $#MinNum#$ & $#BestNum#$ & $#MaxNum#$\\\\
		\\multicolumn{4}{c}{\\includegraphics[width=4\\basewidth]{losses.pdf}}
	\\end{tabular}
\\end{document}
    """
    latex = LATEX_TEMPLATE
    latex = latex.replace("#MinStr#", smoothness_priors[0])
    latex = latex.replace("#BestStr#", smoothness_priors[best_index])
    latex = latex.replace("#MaxStr#", smoothness_priors[-1])
    latex = latex.replace("#MinNum#", "%.3f"%X[0])
    latex = latex.replace("#BestNum#", "%.3f" % X[best_index])
    latex = latex.replace("#MaxNum#", "%.3f" % X[-1])
    with open(os.path.join(output_folder, "comparison.tex"), "w") as f:
        f.write(latex)


def visualize_resolutions(settings_file, name, smoothness_prior, show_or_export: str):
    assert show_or_export in ["show", "export"]

    print("Load results")
    folder = get_result_folder("tf/image_" + name + ("/smooth%05d"%int(smoothness_prior*10000)))
    data = np.load(os.path.join(folder, "data.npz"), allow_pickle=True)
    resolutions = []
    local_results = {}
    for r in data['resolutions']:
        if ('r%d'%r) in data:
            resolutions.append(r)
            local_results[r] = data['r%d'%r].item()
    #resolutions = data['resolutions']
    #local_results = dict([(r, data['r%d'%r].item()) for r in resolutions])
    reference = data['reference'].item()
    NumViews = reference['images'].shape[0]

    s, volume, device, cameras, renderer, render_with_tf = \
        load_settings(settings_file)

    print("Create plots")

    def export_timings():
        keys = ['forward-immediate', 'adjoint-immediate', 'adjoint-delayed']
        keyNames = ['forward-immediate', 'adjoint-immediate', 'adjoint-delayed']
        X = np.arange(len(resolutions))
        CI = 1.96
        show_confidence_interval = False

        # plot
        fig = plt.figure(constrained_layout=True, figsize=(6, 2.5))
        spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[3,1], height_ratios=[1])
        axs_time = fig.add_subplot(spec[0,0])
        axs_loss = fig.add_subplot(spec[0,1])
        axs_time.set_ylabel("Time (s)")
        axs_time.set_xlabel("Resolution")
        axs_time.set_xticks(X)
        axs_time.set_xticklabels(resolutions)
        axs_time.set_yscale('log')

        if 'adjoint-delayed' in local_results[resolutions[0]]:
            first_key = 'adjoint-delayed'
        else:
            first_key = None
            for key in keys:
                if key in local_results[resolutions[0]]:
                    first_key = key
                    break
        forward_only = np.stack([
            local_results[r][first_key]['forward_ms'] / 1000.0 / NumViews
            for r in resolutions], axis=0)
        forward_only_mean = np.mean(forward_only, axis=1)
        forward_only_std = np.std(forward_only, axis=1)
        forward_only_ci = CI * forward_only_std  # / forward_only_mean

        lines_forward_only = axs_time.plot(X, forward_only_mean, '.-', label='forward pass only')[0]
        if show_confidence_interval:
            axs_time.fill_between(
                X, forward_only_mean - forward_only_ci, forward_only_mean + forward_only_ci,
                color=lines_forward_only.get_color(), alpha=.1)

        max_memory = 0
        for key,key_label in zip(keys, keyNames):
            if not (key in local_results[resolutions[0]]):
                print("key", key, "not found")
                continue

            times, memory = [], []
            for r in resolutions:
                if key in local_results[r]:
                    times.append(
                        (local_results[r][key]['forward_ms'] +
                        local_results[r][key]['backward_ms'])/1000.0/NumViews)
                    memory.append(
                        np.mean(local_results[r][key]['forward_bytes']/1024/1024/NumViews))
                else:
                    break
            times = np.stack(times, axis=0)
            times_mean = np.mean(times, axis=1)
            times_std = np.std(times, axis=1)
            times_ci = CI * times_std# / forward_immediate_mean
            #print(key)
            #print(" times:", times_mean)
            #print(" memory:", memory)

            lines = axs_time.plot(X[:len(times_mean)], times_mean, '.-', label=key_label)[0]
            if show_confidence_interval:
                axs_time.fill_between(
                    X[:len(times_mean)], times_mean - times_ci, times_mean + times_ci,
                    color=lines.get_color(), alpha=.1)
            #axs_time[1].plot(X[:len(memory)], memory, color=lines.get_color())
            max_memory = max(max_memory, np.max(memory))

        if show_confidence_interval:
            axs_time.text(0.02,0.95, 'with 95% confidence',
                        horizontalalignment='left', verticalalignment='top',
                        transform = axs_time[0].transAxes)
        #axs_time[0].set_ylim(bottom=0)
        #axs_time.set_ylim(bottom=0, top=max_memory*1.05)

        # loss values
        axs_loss.set_ylabel("LPIPS")
        axs_loss.set_xlabel("Resolution")
        axs_loss.set_xticks(X)
        axs_loss.set_xticklabels(resolutions)
        #axs_loss.set_yscale('log')

        reference_white = blend_to_white(render_with_tf(reference['tf']))
        Ylpips = [None] * len(resolutions)
        for i,r in enumerate(resolutions):
            image_white = blend_to_white(
                render_with_tf(local_results[r]["adjoint-delayed"]['tfs'][-1]))
            Ylpips[i] = lpips(image_white, reference_white)

        renderer_key = 'adjoint-delayed'
        #losses = [local_results[r][renderer_key]['losses'][-1] for r in resolutions]
        #axs_loss.bar(X, losses)
        axs_loss.bar(X, Ylpips)

        handles, labels = axs_time.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2)

        plt.subplots_adjust(top=0.774,bottom=0.173,left=0.117,right=0.991,hspace=0.2,wspace=0.355)
        if show_or_export == "show":
            plt.show()
        else:
            plt.savefig(os.path.join(folder, "timings.pdf"))
            plt.close(fig)
    export_timings()

    def save_img(img, prefix):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(folder, prefix + '.png'), img)

    def export_images():
        renderer_key = 'adjoint-delayed'

        tf_figsize = (7,2)
        tf_to_texture(reference['tf'], os.path.join(folder, "reference-tf.png"))
        save_img(render_with_tf(reference['tf']), "reference-img")
        for r in resolutions:
            tf_to_texture(
                local_results[r][renderer_key]['tfs'][0], os.path.join(folder, "optim-tf-r%02d-initial.png"%r))
            save_img(
                render_with_tf(local_results[r][renderer_key]['tfs'][0]),
                "optim-img-r%02d-initial"%r)
            tf_to_texture(
                local_results[r][renderer_key]['tfs'][-1], os.path.join(folder, "optim-tf-r%02d-final.png" % r))
            save_img(
                render_with_tf(local_results[r][renderer_key]['tfs'][-1]),
                "optim-img-r%02d-final" % r)
        r_last = resolutions[-1]
        for step,tf in enumerate(local_results[r_last][renderer_key]['tfs']):
            tf_to_texture(tf, os.path.join(folder, "optim-tf-r%02d-step%03d.png" % (r_last,step)))
            save_img(
                render_with_tf(tf),
                "optim-img-r%02d-step%03d" % (r_last,step))
    if show_or_export == 'export':
        export_images()

    def export_losses():
        renderer_key = 'adjoint-delayed'
        fig = plt.figure(figsize=(7,5))
        ax = plt.gca()
        for r in resolutions:
            Y = local_results[r][renderer_key]['losses']
            X = np.arange(len(Y))
            ax.plot(X, Y, 'o-', label="R=%d"%r, ms=3)
        ax.legend()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")

        plt.subplots_adjust(0.10, 0.09, 0.98, 0.96)
        if show_or_export == "show":
            plt.show()
        else:
            plt.savefig(os.path.join(folder, "losses.pdf"))
            plt.close(fig)
    export_losses()

def visualize_generalization(scenes: list, smoothness_prior: float, difference_scaling: float):
    for settings_file, name, human_name in scenes:
        print("Load results")
        folder = get_result_folder("tf/image_" + name + ("/smooth%05d" % int(smoothness_prior * 10000)))
        data = np.load(os.path.join(folder, "data.npz"), allow_pickle=True)

        s, volume, device, cameras, renderer, render_with_tf = \
            load_settings(settings_file)

        def save_img(img, prefix):
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(folder, prefix + '.png'), img)

        # find highest resolution
        resolutions = []
        for r in data['resolutions']:
            if ('r%d' % r) in data:
                resolutions.append(r)
        resolutions.sort()
        resolution_to_use = resolutions[-1]

        # fetch TF and render
        renderer_key = 'adjoint-delayed'
        local_result = data['r%d' % resolution_to_use].item()
        initial_tf = local_result[renderer_key]['tfs'][0]
        initial_image = render_with_tf(initial_tf)
        final_tf = local_result[renderer_key]['tfs'][-1]
        final_image = render_with_tf(final_tf)
        reference = data['reference'].item()
        reference_tf = reference['tf']
        reference_image = render_with_tf(reference_tf)

        # save
        tf_to_texture(initial_tf, os.path.join(folder, "%s-tf-initial.png"%human_name))
        save_img(initial_image, "%s-image-initial"%human_name)
        tf_to_texture(final_tf, os.path.join(folder, "%s-tf-final.png"%human_name))
        save_img(final_image, "%s-image-final"%human_name)
        tf_to_texture(reference_tf, os.path.join(folder, "%s-tf-reference.png"%human_name))
        save_img(reference_image, "%s-image-reference"%human_name)

        # blend to white for comparisons
        final_white = blend_to_white(final_image)
        reference_white = blend_to_white(reference_image)

        # stats
        global_ssim = ssim(final_white, reference_white)
        global_psnr = psnr(final_white, reference_white)

        local_ssim = toHWC(losses.ssim.SSIM(size_average="none")(
            toCHW(final_white.unsqueeze(0)), toCHW(reference_white.unsqueeze(0)))).numpy()[0]
        print("local ssim: min=%.5f, max=%.5f"%(np.min(local_ssim), np.max(local_ssim)))
        print("local ssim shape:", local_ssim.shape)
        save_img(local_ssim, "%s-image-ssim"%human_name)

        local_diff = np.clip(1 - difference_scaling * np.abs((final_white-reference_white).numpy()), 0.0, 1.0)
        save_img(local_diff, "%s-image-difference" % human_name)

        with open(os.path.join(folder, "%s-stats.txt"%human_name), "w") as f:
            f.write("PSNR: %.5f dB\n" % global_psnr)
            f.write("SSIM: %.5f\n" % global_ssim)

if __name__ == '__main__':
    iterations = 200
    optimizer = torch.optim.Adam

    # Figure out the best smoothing prior (Figure 7)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.000, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.005, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.002, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.001, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.015, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.010, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.020, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.030, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.040, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.050, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.100, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.200, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.300, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.400, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 0.500, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 1.000, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 2.000, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 3.000, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 4.000, iterations, optimizer, False)
    compute("config-files/skull1b.json", "SkullSmoothing", 5.000, iterations, optimizer, False)
    visualize_smoothingPriors("config-files/skull1b.json", "SkullSmoothing")
    best_smoothing_prior = 0.4

    # Compare algorithms and evaluate resolution (Figure 8, Figure 1b)
    compute("config-files/skull1b.json", "SkullAll", best_smoothing_prior, iterations, optimizer, all_renderer=True, Min_R=2)
    visualize_resolutions("config-files/skull1b.json", "SkullAll", best_smoothing_prior, show_or_export="export")

    # Generalization (Figure 9)
    compute("config-files/thorax4pw.json", "Thorax4pw", best_smoothing_prior, iterations, optimizer, all_renderer=False, Min_R=6)
    compute("config-files/plume123-linear-fancy2.json", "Plume123Fancy", best_smoothing_prior, iterations, optimizer, all_renderer=False, Min_R=6)
    visualize_generalization([
        ("config-files/thorax4pw.json", "Thorax4pw", "Thorax"),
        ("config-files/plume123-linear-fancy.json", "Plume123Fancy", "Plume")
    ], best_smoothing_prior, difference_scaling = 10)
