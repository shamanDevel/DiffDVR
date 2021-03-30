#pragma once

#include "helper_math.cuh"
#include "renderer_commons.cuh"
#include "renderer_tensor.cuh"


//=========================================
// MEMORY LAYOUT
//=========================================
//
// A note on dimensions: if leading dimensions are in brackets,
// they can be of size 1 and are broadcasted.
//
// Density volume: (B) * X * Y * Z  float
//
// Camera, one of:
//  - RayStart:  (B) * H * W * 3   float
//    RayDir:    (B) * H * W * 3  float
//  - EyePos:    (B) * 3
//    ViewRight: (B) * 3
//    ViewUp:    (B) * 3
//
//  BoxMin:  (B)*3  float, not differentiable
//  BoxSize: (B)*3  float, not differentiable
//
//  StepSize: (B) * (H) * (W)  float
//
//  Transfer Function, one of:
//   - TFIdentity:   (B) * 1 * 2,  identity TF with scaling on opacity and color, last dimension
//        the extra dimensions is here so that all TF types can be stored in a 3D-tensor
//   - TFTexture:    (B) * R * 4,  1D texture with resolution R, equal-spaced, linearly interpolated
//        control point. Each control point stores (r,g,b,absorption)
//   - TFLinear:     (B) * R * 5,  R control points of a piecewise linear TF
//        Each control point stores (r,g,b,absorption, position)
//   - TFGaussian:   (B) * R * 6,  R control points of a sum-of-gaussians TF
//        Each control point stores (r,g,b,absorption, mean, var)
//
//
//  Outputs:
//  Color: B * H * W * 4  r,g,b,alpha
//  TerminationIndex: B * H * W  int. Index of early-ray termination
//


//=========================================
// RENDERER SETTINGS
//=========================================

namespace kernel
{
	typedef BroadcastingTensorAccessor32<real_t, 1, RestrictPtrTraits> Tensor1Read;
	typedef BroadcastingTensorAccessor32<real_t, 2, RestrictPtrTraits> Tensor2Read;
	typedef BroadcastingTensorAccessor32<real_t, 3, RestrictPtrTraits> Tensor3Read;
	typedef BroadcastingTensorAccessor32<real_t, 4, RestrictPtrTraits> Tensor4Read;
	typedef BroadcastingTensorAccessor32<real_t, 5, RestrictPtrTraits> Tensor5Read;
	
	typedef PackedTensorAccessor32<real_t, 1, DefaultPtrTraits> Tensor1RW;
	typedef PackedTensorAccessor32<real_t, 2, DefaultPtrTraits> Tensor2RW;
	typedef PackedTensorAccessor32<real_t, 3, DefaultPtrTraits> Tensor3RW;
	typedef PackedTensorAccessor32<real_t, 4, DefaultPtrTraits> Tensor4RW;
	typedef PackedTensorAccessor32<real_t, 5, DefaultPtrTraits> Tensor5RW;

	typedef BroadcastingTensorAccessor32<real_t, 3, DefaultPtrTraits> BTensor3RW;
	typedef BroadcastingTensorAccessor32<real_t, 4, DefaultPtrTraits> BTensor4RW; //Tensor4RW with broadcasting
	
	typedef BroadcastingTensorAccessor32<int, 1, RestrictPtrTraits> ITensor1Read;
	typedef BroadcastingTensorAccessor32<int, 2, RestrictPtrTraits> ITensor2Read;
	typedef BroadcastingTensorAccessor32<int, 3, RestrictPtrTraits> ITensor3Read;
	typedef BroadcastingTensorAccessor32<int, 4, RestrictPtrTraits> ITensor4Read;
	typedef BroadcastingTensorAccessor32<int, 5, RestrictPtrTraits> ITensor5Read;
	
	typedef PackedTensorAccessor32<int, 1, DefaultPtrTraits> ITensor1RW;
	typedef PackedTensorAccessor32<int, 2, DefaultPtrTraits> ITensor2RW;
	typedef PackedTensorAccessor32<int, 3, DefaultPtrTraits> ITensor3RW;
	typedef PackedTensorAccessor32<int, 4, DefaultPtrTraits> ITensor4RW;
	typedef PackedTensorAccessor32<int, 5, DefaultPtrTraits> ITensor5RW;

	typedef BroadcastingTensorAccessor32<int64_t, 1, RestrictPtrTraits> LTensor1Read;

	enum VolumeFilterMode
	{
		FilterNearest,
		FilterTrilinear,
		//FilterTricubic,
		__VolumeFilterModeCount__,
		
		FilterPreshaded, //volume contains pre-shaded colors in the batch dimension
		FilterNetwork, //scene network is used instead of a volume
	};

	enum CameraMode
	{
		/*
		 * camera specified as per-pixel ray start+direction.
		 * See RendererInputs::cameraRayStart and RendererInputs::cameraRayDir
		 */
		CameraRayStartDir,
		/**
		 * camera specified as inverse view-projection matrix.
		 * See RendererInputs::cameraMatrix as B*4*4 matrix
		 */
		CameraInverseViewMatrix,
		/**
		 * camera specified as 3x3 matrix (see Camera::generateRays for details) plus fovY.
		 * See RendererInputs::cameraMatrix as B*3*3 matrix where
		 *  - cameraMatrix[:,0,:] is the camera/eye position,
		 *  - cameraMatrix[:,1,:] the right vector and
		 *  - cameraMatrix[:,2,:] the up vector,
		 *  
		 * Together with RendererInputs::cameraFovYRadians
		 */
		CameraReferenceFrame,
		///**
		// * Camera specified as position on a sphere.
		// * See RendererInputs::cameraMatrix as B*2*3 matrix where
		// *  - cameraMatrix[:,0,0] = yaw (radians)
		// *  - cameraMatrix[:,0,1] = pitch (radians)
		// *  - cameraMatrix[:,0,2] = distance
		// *  - cameraMatrix[:,1,:] = center
		// *  
		// * Together with RendererInputs::cameraFovYRadians
		// */
		//CameraSphere, //camera specified as cameraMatrix
		__CameraModeCount__,
	};

	enum TFMode
	{
		TFIdentity, //identity TF, only scaling on absorption and color
		TFTexture,  //1D texture
		TFLinear,   //piecewise linear TF
		TFGaussian, //sum-of-gaussians
		__TFModeCount__,

		//"Hidden" TF modes, that is, hidden from the GUI and used internally
		TFPreshaded, //together with VolumeFilterMode::FilterPreshaded, passes pre-shaded colors on
		TFGaussianLog, //sum-of-gaussians, but the logarithm is returned
	};

	enum BlendMode
	{
		BlendBeerLambert, // alpha = 1 - exp(-absorption*stepsize)
		BlendAlpha,       // alpha = min(1, absorption*stepsize)
		__BlendModeCount__
	};

	//Inputs to the forward passes
	//These are the general parameters, some differentiable, some not
	struct RendererInputs
	{
		int2 screenSize; //W*H
		                 //must match the sizes of per-pixel settings

		//VolumeFilterMode volumeFilterMode; //-> template parameter
		Tensor4Read volume; //B*X*Y*Z
		int3 volumeSize; //X*Y*Z for faster access

		Tensor2Read boxMin; //B*3
		Tensor2Read boxSize; //B*3

		//CameraMode cameraMode; //-> template parameter
		Tensor4Read cameraRayStart; //B*H*W*3
		Tensor4Read cameraRayDir; //B*H*W*3
		/**
		 * cameraMode==CameraInverseViewMatrix -> inverse view-projection matrix
		 *  of shape B*4*4
		 * cameraMode==CameraReferenceFrame -> reference frame matrix
		 *  of shape B*3*3
		 */
		Tensor3Read cameraMatrix;
		real_t cameraFovYRadians; //only for cameraMode==CameraReferenceFrame

		Tensor3Read stepSize; //B*H*W

		//TFMode tfMode; //-> template parameter
		Tensor3Read tf; //B*R*C

		real_t blendingEarlyOut = real_t(1) - real_t(1e-5);
	};

	//The outputs of the forward pass
	struct RendererOutputs
	{
		Tensor4RW color; //B*H*W*4 (r,g,b,alpha)
		ITensor3RW terminationIndex; //B*H*W integer
	};

	struct ForwardDifferencesSettings
	{
		static constexpr const int NOT_ENABLED = -1; //the variable is not enabled

		//index into the derivative array for the step size
		int d_stepsize = NOT_ENABLED;

		//derivative indices for the start position of the ray
		int3 d_rayStart = make_int3(NOT_ENABLED, NOT_ENABLED, NOT_ENABLED);
		//derivative indices for the ray direction
		int3 d_rayDir = make_int3(NOT_ENABLED, NOT_ENABLED, NOT_ENABLED);

		//derivative index for the TF parameters, shape B*R*C
		ITensor3Read d_tf;
		bool hasTFDerivatives = false;
	};
	
	//gradient output for forward differences
	//These are the derivatives with respect to the color,
	//hence they are of type real4 / 4 channels
	struct ForwardDifferencesOutput
	{
		//B*H*W*D*4
		//The D-dimension indicates which variable is derived for
		//see 
		Tensor5RW gradients; 
		
	};

	//The outputs of the forward pass as input to the adjoint code
	struct RendererOutputsAsInput
	{
		Tensor4Read color; //B*H*W*4 (r,g,b,alpha)
		ITensor3Read terminationIndex; //B*H*W integer
	};

	typedef Tensor4Read AdjointColor_t;

	//The adjoint variables for the input parameters.
	//As they are written with atomics, they must be initialized with zeros.
	struct AdjointOutputs
	{
		BTensor4RW adj_volume; // (B)*X*Y*Z

		//true iff one of the dimensions of adj_stepSize are broadcasted.
		//Then atomics are used instead of a simple memory write
		bool stepSizeHasBroadcasting;
		BTensor3RW adj_stepSize; // (B)*(H)*(W)

		//True iff the camera arguments have broadcasting over the batches
		bool cameraHasBroadcasting;
		BTensor4RW adj_cameraRayStart; // (B)*H*W*3
		BTensor4RW adj_cameraRayDir; // (B)*H*W*3

		BTensor3RW adj_tf; // (B)*R*C
	};
}
