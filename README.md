# Differentiable Direct Volume Rendering

The source code for the VIS submission 1136 "Differentiable Direct Volume Rendering"

## Requirements

This is the setup which was used to develop the code and execute the experiments. Other setups *should* be supported as well

- Ubuntu 20.0 (only renderer), Windows 10 (renderer+GUI)
- CUDA 11.0
- OpenGL with GLFW (GUI only) and GLM
- Python > 3.6, PyTorch 1.8
  See `environment.yml` or `requirements.txt` for the used packages

Installation: simply run cmake and compile



## Project structure:

### renderer

A static library containing the actual code for the normal, non-differentiated volume renderer, the forward differentiation and the adjoint differentiation.

Currently supported are:

- Camera: Specified as view frustrum (position, right, up), on a sphere (center, longitude, latitude, radius, up), view-projection matrix, or per-pixel ray position and direction
- Volume Interpolation modes: nearest neighbor or trilinear
- Transfer Functions: Identity (the density is simply mapped to color and absorption via a scaling factor), piecewise linear (defined by control points), 1D texture, Sum of Gaussians [1] or None (for pre-shaded color volumes)
- Blending: Beer-Lambert (with exponential function), Alpha (Taylor approximation)

A full user manual is in work, noteworthy files up to then include:

- All files ending in *.cuh: These are the CUDA kernel implementations. See renderer_kernels.cuh for the entry points

### bindings

Exposes a python extension library "pyrenderer" that allows to embed the renderer in python training and evaluation scripts. All experiments are scripted in python.
See bindings.cpp for all exposed functions

### gui

An interactive GUI, Windows only, to explore the datasets and specify the configuration. The usual pipeline for the experiments is as following: load the datasets, modify the camera to find a test pose for evaluation, design the transfer function and save the settings. These setting files are then used as ground truth in the experiments

### pytests

The python scripts for the experiments. See below on how to reproduce the results.

Also contain the config files for the experiments (can be loaded in the GUI) and the datasets



## Reproduce the results

All scripts should be executed with `pytests` as the current working directory.

### Best Viewpoint Selection with an Entropy Measure (Figure 1a, Figure 5, Figure 6)
The script contains it all.

    > python tests/camera/run_entroy2d.py

### Transfer Function Reconstruction
This script contains it all, Figure 1b, Figure 7,8,9

    > python tests/tf/run_reconstruction.py

Older versions:
For reconstructions with a direct control over the hyperparameters via the command line:

    > python tests/tf/train_meta.py \
        results/tf/meta/skull1.hdf5 \
        config-files/skull1.json \
        --views 8 --imgres 256 \
        --tfres 64 --tfmode texture \
        -l1 1 -ps 0.01
        --iterations 200 -lr 0.8 --seed 124

Visualize the result and export images with

    > python tests/tf/vis_meta.py

Style transfer (removed from the paper due to space limitations)

    > python tests/tf/train_styletransfer.py \
        results/tf/style/skull-s42.hdf5 \
        config-files/skull2.json \
        config-files/reference-jet.png \
        --views 4 --imgres 256 \
        --tfres 64 --tfmode texture \
        --minDensity 0.075 \
        --cropnumber 4 --cropsize 64 \
        -ps 0.05
        --styleLayers "conv_5:1,conv_7:1,conv_9:1" \
        --iterations 200 -lr 0.8 --batches 2 \
        --seed 42

The same is repeated for seed 43.
Visualize the result and export images with

    > python tests/tf/vis_styletransfer.py

### Stepsize optimization
(removed from the paper due to space limitations)

    > python tests/stepsize/train_stepsize.py \
        results/stepsize/skullplume/skull.hdf5 \
        config-files/skull4gauss.json \
        -tf gauss -v 1 -m 1 -lr 1.2 --ignoreEmpty
    > python tests/stepsize/train_stepsize.py \
        results/stepsize/skullplume/plume.hdf5 \
        config-files/plume123-linear-fancy2.json \
        -tf linear -v 1 -m 5 -lr 1.2 -s 2 --blendWhite --ignoreEmpty

Visualize with

    > python tests/stepsize/vis_stepsize.py

### Volume Optimization - Absorption-only

The comparisons to ASTRA and Mitsuba are a bit tricky as they have to be installed separately.

For ASTRA, pre-built binaries are available. For example, I placed the ASTRA binaries [2] into the subfolder `tests/volume/astra/astra` and created a conda environment with the necessary dependencies. Then in `tests/volume/compare_reconstruction.py` in line 209 I call the optimization script `tests/volume/astra/VolumeReconstruction.py`. The paths and environments might be different on other systems.

For Mitsuba, clone the "Radiance Backpropagation" repository and the scene data provided on their webpage [3] into the common parent folder `MITSUBA_ROOT`. Replace `MITSUBA_ROOT/mitsuba2/src/media/heterogeneous_absorptive.cpp` with the version provided in `tests/volume/mitsuba/heterogeneous_absorptive.cpp`. Compile the library. Copy `tests/volume/mitsuba/optimize_rb2.py` to `MITSUBA_ROOT/mitsuba2/src/optix/tests/optimize_rb2.py`.

The base command lines for the comparison of reconstructions are
    > python3 compare_reconstruction.py results/volume/density/skull7absorption config-files/skull7absorption.json --views 64 --diffdvrL1 --visCropSlice 73:2:64:96 --visCropRendering 62:250:192:128 --visRenderingDiffScaling 20 --visSliceDiffScaling 5 --visSliceRotate 3
    > python3 compare_reconstruction.py results/volume/density/plume123absorption config-files/plume123-linear-absorption.json --views 64 --diffdvrL1 --visCropSlice 95:125:96:64 --visCropRendering 90:30:192:128 --visRenderingDiffScaling 20 --visSliceDiffScaling 5 --visSliceRotate 2
    > python3 compare_reconstruction.py results/volume/density/thorax2absorption config-files/thorax2absorption.json --views 64 --diffdvrL1 --visSliceIndividualNormalize --visCropSlice 104:37:96:64 --visCropRendering 30:215:192:128 --visRenderingDiffScaling 20 --visSliceDiffScaling 5 --visSliceRotate 0

First, call those scripts with the additional command line argument `-amd`.
`-a`: run ASTRA reconstruction
`-m`: prepare Mitsuba
`-d`: run DiffDVR reconstruction

For the Mitsuba reconstructions, follow the following steps:
1. Copy/Replace from the output folder the file `config.py` to `MITSUBA_ROOT/mitsuba2/src/optix/tests/config.py`
2. Copy from the output folder the file `scene.xml` to `MITSUBA_ROOT/scenes/<SOME_NAME>/scene.xml` (path must match the one in config.py)
3. Copy from the output folder `mitsuba_initial.vol` and `mitsuba_reference.vol` to the sam escene folder as in 2.
4. Run from the build-directory of Mitsuba: `python3 ../src/optix/tests/optimize_rb2.py <name-of-the-scene-as-in-the-config.py>`
5. The above step creates a dedicated output folder in `MITSUBA_ROOT/outputs`. From that output copy `volumes.npz` to the output folder of the DiffDvr-script and rename to `mitsuba-output.npz`. There should already be a `astra-output.npz` and `diffdvr-output.npz`

Create the visualizations (Figure 10) by running the `compare_reconstruction.py`-scripts as above with the additional option `-v`

### Volume Optimization - Emission-Absorption with Transfer Function

Non-monotonic, colored TFs (Figure 11)

    > python tests/volume/train_volume.py \
        results/volume/density\tooth3gauss-direct.hdf5 \
        config-files\tooth3gauss.json \
        -v 64 -tf gauss -R 256 --multiscale 8 -I gauss \
        -l1 1 -ps 5.0 -bm stochastic -lr 0.3 -i 50 \
        --volumeSaveFrequency 5
    > python ./tests/volume/train_volume.py \
        results/volume/density/tooth3gauss-pre.hdf5 \
        ./config-files/tooth3gauss.json \
        --preshaded \
        -v 64 -tf gauss -R 256 --multiscale 8 -I gauss \
        -l1 1 -ps 0.0 -bm stochastic -b 4 -lr 0.3 -i 10 \
        --volumeSaveFrequency 1 --memorySaving
    > python tests/volume/preshaded_to_density.py \
        results/volume/density\tooth3gauss-pre.hdf5 \
        results/volume/density\tooth3gauss-fit.hdf5 \
        -b 1 --smoothOnly -I best-fit -ow -1 -fi 200 -nw 1 -s 1 \
        -i 0 --volumeSaveFrequency 10
    > python tests/volume/train_volume.py \
        results/volume/density\tooth3gauss-recon1.hdf5 \
            config-files\tooth3gauss.json \
            -v 128 -tf gauss -R 256 -I file \
            --initialFilePath results/volume/density\tooth3gauss-fit.hdf5 \
            --initialFileEpoch 18 \
            -l1 1 -ps 20.0 -bm stochastic -lr 0.3 \
            -i 50 --volumeSaveFrequency 5
    
    > python tests/volume/train_volume.py \
        results/volume/density\thorax2gauss256-direct.hdf5 \
        config-files\thorax2gauss256.json \
        -v 64 -tf gauss -R 256 --multiscale 8 -I gauss \
        -l1 1 -ps 5.0 -bm stochastic -lr 0.3 -i 50 \
        --volumeSaveFrequency 5
    > python ./tests/volume/train_volume.py \
        results/volume/density/thorax2gauss256-pre.hdf5 \
        ./config-files/thorax2gauss256.json \
        --preshaded \
        -v 64 -tf gauss -R 256 --multiscale 8 -I gauss \
        -l1 1 -ps 0.0 -bm stochastic -b 4 -lr 0.3 \
        -i 10 --volumeSaveFrequency 1 --memorySaving
    > python tests/volume/preshaded_to_density.py \
        results/volume/density\thorax2gauss256-pre.hdf5 \
        results/volume/density\thorax2gauss256-fit.hdf5 \
        -b 1 --smoothOnly -I best-fit -ow -1 -fi 200 -nw 1 -s 1 \
        -i 200 --volumeSaveFrequency 10
    > python tests/volume/train_volume.py \
        results/volume/density\thorax2gauss256-recon-tiny.hdf5 \
        config-files\thorax2gauss256.json \
        -v 64 -tf gauss -R 256 -I file \
        --initialFilePath results/volume/density\thorax2gauss256-fit.hdf5 \
        --initialFileEpoch 22 \
        -l1 1 -ps 5.0 -bm stochastic -lr 0.3 -i 50 \
        --volumeSaveFrequency 5

Image export, slices, statistics and the LaTeX layouting (Figure 11) is done in

    python tests/volume/export_trained_volumes.py

The 1D test example (Figure 12) is generated by the unit tests, `unittests/testNonCovnexDensity.cpp`.



References

[1] Kniss et al., Gaussian transfer functions for multi-field volume visualization, VIS 2003

[2] http://www.astra-toolbox.com/

[3] http://rgl.epfl.ch/publications/NimierDavid2020Radiative