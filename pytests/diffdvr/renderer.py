import torch
import torch.nn as nn
import numpy as np
import time

from diffdvr.utils import implies
import pyrenderer

class ProfileRenderer:
    def __init__(self):
        self.forward_ms = 0.0
        self.forward_bytes = 0
        self.backward_ms = 0.0

class Timer:
    def __init__(self, enable, cuda):
        self._enable = enable
        self._cuda = cuda
        self._elapsed_ms = None
        if self._cuda:
            self._gpu_timer = pyrenderer.GpuTimer()

    def elapsed_ms(self):
        assert self._elapsed_ms is not None, "No timings recorded"
        return self._elapsed_ms

    def __enter__(self):
        if not self._enable: return
        if self._cuda:
            pyrenderer.sync()
            self._gpu_timer.start()
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._enable: return
        if self._cuda:
            self._gpu_timer.stop()
            pyrenderer.sync()
        stop = time.time()
        if self._cuda:
            self._elapsed_ms = self._gpu_timer.elapsed_ms()
            #elapsed_ms_cpu = (stop - self._start) * 1000.0
            #print("gpu: %.4f, cpu: %.4f"%(self._elapsed_ms, elapsed_ms_cpu))
        else:
            self._elapsed_ms = (stop - self._start) * 1000.0

def size_of_tensors(*tensors_or_list):
    if isinstance(tensors_or_list, (list, tuple)):
        if len(tensors_or_list)>1:
            return sum([size_of_tensors(t) for t in tensors_or_list])
        else:
            tensors_or_list = tensors_or_list[0]
    if tensors_or_list is None: return 0
    assert isinstance(tensors_or_list, torch.Tensor)
    size = tensors_or_list.element_size()
    for s in tensors_or_list.shape:
        size *= s
    return size

class _RendererFunction(torch.autograd.Function):
    @staticmethod
    def __assembleForwardDifferencesSettings(optimize_camera, optimize_tf, optimize_volume, tf):
        s = pyrenderer.ForwardDifferencesSettings()
        D = 0
        if optimize_camera:
            s.d_rayStart = pyrenderer.int3(D, D+1, D+2)
            s.d_rayDir = pyrenderer.int3(D+3, D+4, D+5)
            D += 6
        if optimize_tf:
            s.has_tf_derivatives = True
            B,R,C = tf.shape
            d_tf = np.arange(D, D+R*C, dtype=np.int32)
            d_tf = np.reshape(d_tf, (1, R, C))
            s.d_tf = torch.from_numpy(d_tf).to(device=tf.device)
            D += R*C
        if optimize_volume:
            raise ValueError("forward mode does not support volume derivatives")
        s.D = D
        return s

    @staticmethod
    def forward(ctx,
                camera_matrix : torch.Tensor, camera_ray_start : torch.Tensor,
                camera_ray_dir : torch.Tensor, tf : torch.Tensor, volume : torch.Tensor,
                inputs : pyrenderer.RendererInputs, fov_y_radians : float,
                optimize_camera, optimize_tf, optimize_volume,
                use_adjoint_method, forward_delayed_gradients, tf_delayed_accumulation,
                profiling : ProfileRenderer):
        # save meta-parameters
        ctx.inputs = inputs
        ctx.fov_y_radians = fov_y_radians
        ctx.optimize_camera = optimize_camera
        ctx.optimize_tf = optimize_tf
        ctx.optimize_volume = optimize_volume
        ctx.use_adjoint_method = use_adjoint_method
        ctx.forward_delayed_gradients = forward_delayed_gradients
        ctx.tf_delayed_accumulation = tf_delayed_accumulation
        ctx.profiling = profiling

        # assemble input
        assert implies(optimize_camera, inputs.camera_mode == pyrenderer.CameraMode.RayStartDir)
        if inputs.camera_mode == pyrenderer.CameraMode.RayStartDir:
            inputs.camera = pyrenderer.CameraPerPixelRays(camera_ray_start, camera_ray_dir)
        else:
            inputs.camera = pyrenderer.CameraReferenceFrame(camera_matrix, fov_y_radians)
        inputs.tf = tf
        inputs.volume = volume

        # allocate outputs
        tensors_for_batches = [camera_matrix, camera_ray_start, camera_ray_dir, tf]
        if inputs.volume_filter_mode != pyrenderer.VolumeFilterMode.Preshaded:
            tensors_for_batches.append(volume)
        B = max([(1 if t is None else t.shape[0]) for t in tensors_for_batches])
        W = inputs.screen_size.x
        H = inputs.screen_size.y
        output_color = torch.empty(
            B, H, W, 4, dtype=volume.dtype, device=volume.device)
        output_termination_index = torch.empty(
            B, H, W, dtype=torch.int32, device=volume.device)
        outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)

        # render
        if use_adjoint_method:
            # adjoint method
            with Timer(profiling is not None, volume.is_cuda) as timer:
                pyrenderer.Renderer.render_forward(inputs, outputs)
            if profiling is not None:
                profiling.forward_ms += timer.elapsed_ms()
                profiling.forward_bytes += size_of_tensors(
                    camera_matrix, camera_ray_start, camera_ray_dir, tf, volume, output_color, output_termination_index)
            ctx.save_for_backward(camera_matrix, camera_ray_start, camera_ray_dir, tf, volume, output_color, output_termination_index, None)
        else:
            # forward method
            if forward_delayed_gradients:
                # re-do forward pass in the backward method, then with gradients.
                # For now, just a simple forward pass but we don't need to save the outputs
                with Timer(profiling is not None, volume.is_cuda) as timer:
                    pyrenderer.Renderer.render_forward(inputs, outputs)
                if profiling is not None:
                    profiling.forward_ms += timer.elapsed_ms()
                    profiling.forward_bytes += size_of_tensors(
                        camera_matrix, camera_ray_start, camera_ray_dir, tf, volume)
                ctx.save_for_backward(camera_matrix, camera_ray_start, camera_ray_dir, tf, volume, None, None, None)
            else:
                differences_settings = _RendererFunction.__assembleForwardDifferencesSettings(
                    optimize_camera, optimize_tf, optimize_volume, tf)
                ctx.differences_settings = differences_settings
                gradients_out = torch.zeros(B, H, W, differences_settings.D, 4,
                                            dtype=volume.dtype, device=volume.device)
                with Timer(profiling is not None, volume.is_cuda) as timer:
                    pyrenderer.Renderer.render_forward_gradients(
                        inputs, differences_settings, outputs, gradients_out)
                if profiling is not None:
                    profiling.forward_ms += timer.elapsed_ms()
                    profiling.forward_bytes += size_of_tensors(
                        camera_matrix, camera_ray_start, camera_ray_dir, tf)
                ctx.save_for_backward(camera_matrix, camera_ray_start, camera_ray_dir, tf, None, None, None, gradients_out)

        return output_color

    @staticmethod
    def backward(ctx, grad_output_color):
        grad_camera_matrix, grad_camera_ray_start, grad_camera_ray_dir, grad_tf, grad_volume = \
            None, None, None, None, None

        # restore values
        camera_matrix, camera_ray_start, camera_ray_dir, tf, volume, \
        output_color, output_termination_index, forward_gradients_out = \
            ctx.saved_tensors
        inputs = ctx.inputs
        fov_y_radians = ctx.fov_y_radians
        optimize_camera = ctx.optimize_camera
        optimize_tf = ctx.optimize_tf
        optimize_volume = ctx.optimize_volume
        use_adjoint_method = ctx.use_adjoint_method
        forward_delayed_gradients = ctx.forward_delayed_gradients
        tf_delayed_accumulation = ctx.tf_delayed_accumulation
        profiling : ProfileRenderer = ctx.profiling

        B, W, H, _ = grad_output_color.shape

        if use_adjoint_method:
            # assemble input
            if inputs.camera_mode == pyrenderer.CameraMode.RayStartDir:
                inputs.camera = pyrenderer.CameraPerPixelRays(camera_ray_start, camera_ray_dir)
            else:
                inputs.camera = pyrenderer.CameraReferenceFrame(camera_matrix, fov_y_radians)
            inputs.tf = tf
            inputs.volume = volume

            # construct adjoint outputs
            gradients_out = pyrenderer.AdjointOutputs()
            if optimize_camera:
                grad_camera_ray_start = torch.zeros_like(camera_ray_start)
                grad_camera_ray_dir = torch.zeros_like(camera_ray_dir)
                gradients_out.has_camera_derivatives = True
                gradients_out.adj_camera_ray_start = grad_camera_ray_start
                gradients_out.adj_camera_ray_dir = grad_camera_ray_dir
            if optimize_tf:
                grad_tf = torch.zeros_like(tf)
                gradients_out.has_tf_derivatives = True
                gradients_out.tf_delayed_accumulation = tf_delayed_accumulation
                gradients_out.adj_tf = grad_tf
            if optimize_volume:
                grad_volume = torch.zeros_like(volume)
                gradients_out.has_volume_derivatives = True
                gradients_out.adj_volume = grad_volume

            # adjoint rendering
            outputs_from_forward = pyrenderer.RendererOutputs(output_color, output_termination_index)
            with Timer(profiling is not None, volume.is_cuda) as timer:
                pyrenderer.Renderer.render_adjoint(
                    inputs, outputs_from_forward, grad_output_color, gradients_out)
            if profiling is not None:
                profiling.backward_ms += timer.elapsed_ms()

        else:
            # forward method
            if forward_delayed_gradients:
                # re-do forward pass, now with gradients
                output_color = torch.empty(
                    B, H, W, 4, dtype=volume.dtype, device=volume.device)
                output_termination_index = torch.empty(
                    B, H, W, dtype=torch.int32, device=volume.device)
                outputs = pyrenderer.RendererOutputs(output_color, output_termination_index)
                differences_settings = _RendererFunction.__assembleForwardDifferencesSettings(
                    optimize_camera, optimize_tf, optimize_volume, tf)
                forward_gradients_out = torch.zeros(B, H, W, differences_settings.D, 4,
                                                    dtype=volume.dtype, device=volume.device)
                with Timer(profiling is not None, volume.is_cuda) as timer:
                    pyrenderer.Renderer.render_forward_gradients(
                        inputs, differences_settings, outputs, forward_gradients_out)
                if profiling is not None:
                    profiling.backward_ms += timer.elapsed_ms()
            else:
                # gradients were already computed in the forward pass
                differences_settings = ctx.differences_settings

            # now combine gradients_out with grad_output_color
            # and write the results in grad_camera_ray_start,  grad_camera_ray_dir and grad_tf
            gradients_out = pyrenderer.AdjointOutputs()
            if optimize_camera:
                grad_camera_ray_start = torch.zeros_like(camera_ray_start)
                grad_camera_ray_dir = torch.zeros_like(camera_ray_dir)
                gradients_out.has_camera_derivatives = True
                gradients_out.adj_camera_ray_start = grad_camera_ray_start
                gradients_out.adj_camera_ray_dir = grad_camera_ray_dir
            if optimize_tf:
                grad_tf = torch.zeros_like(tf)
                gradients_out.has_tf_derivatives = True
                gradients_out.adj_tf = grad_tf
            with Timer(profiling is not None, forward_gradients_out.is_cuda) as timer:
                pyrenderer.Renderer.forward_variables_to_gradients(
                    forward_gradients_out, grad_output_color, differences_settings, gradients_out)
            if profiling is not None:
                profiling.backward_ms += timer.elapsed_ms()


        return grad_camera_matrix, grad_camera_ray_start, grad_camera_ray_dir, \
               grad_tf, grad_volume, None, None, None, None, None, None, None, None, None


class Renderer(nn.Module):

    def __init__(self,
                 settings : pyrenderer.RendererInputs = None,
                 optimize_camera = False, optimize_tf = False, optimize_volume = False,
                 gradient_method = "adjoint", forward_delayed_gradients = False,
                 tf_delayed_accumulation = True):
        """
        Constructs a new renderer module.
        :param settings: the renderer settings (pyrenderer.RendererInputs).
          If None, a new instance is created.
          The settings can be accessed via the property 'settings'
        :param optimize_camera: True if the camera parameters (specified via the 3x3 view matrix)
          should be optimized. The view matrix must then be passed to the forward method.
        :param optimize_tf: True if the transfer function should be optimized
        :param optimize_volume: True if the volume densities should be optimized
        :param gradient_method: The method on how the gradients should be computed.
          Can be "adjoint" or "forward"
        :param forward_delayed_gradients: For "forward" mode only:
          If False (default), the forward derivatives are already computed
            in the regular forward pass (faster, more memory).
          If True, the forward derivatives are computed by another rendering step in
            the backward pass (slower, less memory)
        """
        super().__init__()
        self._settings = settings if settings is not None else pyrenderer.RendererInputs()
        self._optimize_camera = optimize_camera
        self._optimize_tf = optimize_tf
        self._optimize_volume = optimize_volume
        self._has_parameters_to_optimize = optimize_camera or optimize_tf or optimize_volume
        assert gradient_method in ["adjoint", "forward"]
        self._use_adjoint_method = gradient_method == "adjoint"
        self._forward_delayed_gradients = forward_delayed_gradients
        self._tf_delayed_accumulation = tf_delayed_accumulation
        assert implies(optimize_volume, gradient_method == "adjoint"), \
            "if volume densities should be optimized, the gradient method must be 'adjoint'"

        self._renderer_function = _RendererFunction.apply

    def get_settings(self):
        return self._settings
    def set_settings(self, settings : pyrenderer.RendererInputs):
        assert settings is not None
        self._settings = settings
    settings = property(get_settings, set_settings)

    def train(self, mode: bool = True):
        if mode is True and not self._has_parameters_to_optimize:
            raise ValueError("requested training mode, but no parameters were specified to be optimized")
        return super().train(mode)

    def forward(self, *,
                camera : torch.Tensor = None, fov_y_radians : float = None,
                tf : torch.Tensor = None, volume : torch.Tensor = None,
                profiling : ProfileRenderer = None):
        """
        Renders the volume
        :param camera: camera reference frame, shape B*3*3
        :param fov_y_radians: the camera field-of-view along the y-axis in radians
        :param tf: transfer function, B*R*C
        :param volume: volume densities, B*X*Y*Z
        :param profiling: if not None, timings for the forward and backward pass in milliseconds are added in here
        :return: the color image, B*H*W*3
        """
        # check if arguments are available
        assert implies(self._optimize_camera, camera is not None)
        assert implies(self._optimize_camera, fov_y_radians is not None)
        assert implies(self._optimize_tf, tf is not None)
        assert implies(self._optimize_volume, volume is not None)

        # clone RendererInputs()
        inputs = self._settings.clone()

        camera_matrix, camera_ray_start, camera_ray_dir = None, None, None
        if camera is not None:
            if self._optimize_camera:
                camera_ray_start, camera_ray_dir = pyrenderer.Camera.generate_rays(
                    camera, fov_y_radians, self._settings.screen_size.x, self._settings.screen_size.y)
                inputs.camera_mode = pyrenderer.CameraMode.RayStartDir
            else:
                camera_matrix = camera
                inputs.camera_mode = pyrenderer.CameraMode.ReferenceFrame

        if tf is None:
            tf = inputs.tf

        if volume is None:
            volume = inputs.volume

        # render
        return self._renderer_function(
            camera_matrix, camera_ray_start, camera_ray_dir, tf, volume,
            inputs, fov_y_radians,
            self._optimize_camera, self._optimize_tf, self._optimize_volume,
            self._use_adjoint_method, self._forward_delayed_gradients, self._tf_delayed_accumulation,
            profiling)
