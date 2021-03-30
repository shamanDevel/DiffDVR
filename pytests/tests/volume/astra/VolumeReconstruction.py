import astra
import numpy as np
import imageio
import argparse
import time

def _main():
    parser = argparse.ArgumentParser(
        description='ASTRA optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=str)
    parser.add_argument("--input",type=str)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--iterations", type=int)
    opt = parser.parse_args()

    npzfile = np.load(opt.input)
    volume = npzfile["volume"][0]
    cameras = npzfile["cameras"]
    fov_y_radians = npzfile["fov_y_radians"]
    world_size = npzfile["world_size"]
    resolution = opt.resolution
    iterations = opt.iterations
    print("volume:", volume.shape)
    print("cameras:", cameras.shape)
    print("fov_y_radians:", fov_y_radians)
    print("world_size:", world_size)

    res = volume.shape # (x, y, z)
    num_cameras = cameras.shape[0]
    # cameras[batch, 0, :] = eye position (x,y,z)
    # cameras[batch, 1, :] = right vector (x,y,z), normalized
    # cameras[batch, 2, :] = up vector (x,y,z), normalized
    # -->
    # const real_t fx = 2 * (x + 0.5f) / real_t(inputs.screenSize.x) - 1; //NDC in [-1,+1]
    # const real_t fy = 2 * (y + 0.5f) / real_t(inputs.screenSize.y) - 1;
    # real3 dir = normalize(cross(up,right) + fx * tanFovX * right + fy * tanFovY * up);

    # Create a 3D volume geometry.
    # Parameter order: rows, colums, slices (y, x, z)
    vol_geom = astra.create_vol_geom(res[1], res[0], res[2])


    # Create volumes

    # initialized to zero
    v0 = astra.data3d.create('-vol', vol_geom)

    # initialized to 3.0
    v1 = astra.data3d.create('-vol', vol_geom, 3.0)

    # initialized to a matrix. A may be a single or double array.
    # Coordinate order: slice, row, column (z, y, x)
    A = np.moveaxis(volume, [0,1,2], [2,1,0])
    #v2 = astra.data3d.create('-vol', vol_geom, A)

    # cone_vec camera transformation
    det_row_count = resolution
    det_col_count = resolution
    astra_cameras = np.zeros((num_cameras, 12))
    world_to_object = np.max(world_size * res) / 2
    print("world to object scaling:", world_to_object)
    world_to_object = np.array([world_to_object]*3)
    fov_x_radians = fov_y_radians * det_col_count / det_row_count
    tan_fov_x = np.tan(fov_x_radians)
    tan_fov_y = np.tan(fov_y_radians)
    for batch in range(num_cameras):
        # astra_cameras[batch,:] = ( srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ )
        # src : the ray source
        # d : the center of the detector
        # u : the vector from detector pixel (0,0) to (0,1)  (det_col_count)
        # v : the vector from detector pixel (0,0) to (1,0)  (det_row_count)
        # direction for ray i=row, j=col: s + (i+0.5)*v + (j+0.5)*u - src
        #   with s = d - 0.5*det_row_count*v - 0.5*det_col_count*u  (center to corner)
        src = cameras[batch,0,:] * world_to_object * 1.7
        right = cameras[batch,1,:]
        up = cameras[batch,2,:]
        front = np.cross(up, right)
        d = src + front
        v = (tan_fov_y / det_row_count) * up
        u = (tan_fov_x / det_col_count) * right
        astra_cameras[batch,:] = np.concatenate([src, d, u, v])
    print(astra_cameras)

    # Render images
    angles = np.linspace(0, np.pi, num_cameras,False)
    #proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, 256, 256, angles, 0, 300)
    #proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, det_row_count, det_col_count, angles, 1000, 0)
    #proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, det_row_count, det_col_count, angles)
    proj_geom = astra.create_proj_geom('cone_vec', det_row_count, det_col_count, astra_cameras)

    # Create projection data from this
    proj_id, proj_data = astra.create_sino3d_gpu(A, proj_geom, vol_geom)
    # proj_data is of shape (det_row_count, num_cameras, det_col_count)
    print("proj_id:", proj_id)
    print("proj_data:", proj_data.shape)
    input_images = proj_data[...]

    # RECONSTRUCTION
    rec_id = astra.data3d.create('-vol', vol_geom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    #cfg['option.DetectorSuperSampling'] = 2 # does not work, is not recognized
    #cfg['option.VoxelSuperSampling'] = 1

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    # Run 150 iterations of the algorithm
    # Note that this requires about 750MB of GPU memory, and has a runtime
    # in the order of 10 seconds.
    print("Run reconstruction")
    tstart = time.time()
    astra.algorithm.run(alg_id, iterations)
    tend = time.time()
    print("Done in", (tend-tstart), "s")

    # Get the result
    rec = astra.data3d.get(rec_id)
    print("result:", rec.shape, ", min=", np.min(rec), ", max=", np.max(rec))

    # render
    print("Render again")
    proj_id, proj_data = astra.create_sino3d_gpu(rec, proj_geom, vol_geom)
    output_images = proj_data[...]

    # save
    print("save")
    rec_out = np.moveaxis(rec, [0, 1, 2], [2, 1, 0])
    np.savez(opt.output,
             volume=rec_out,
             input_images=np.moveaxis(input_images, [0,1,2], [1,0,2]),
             output_images=np.moveaxis(output_images, [0,1,2], [1,0,2]),
             time_sec = (tend-tstart))



if __name__ == '__main__':
    _main()