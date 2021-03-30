"""
Creates the script files for train_meta.py
"""

import os
import itertools

def create_files(output_dir, script_dir, launcher_file):
    settings_files = ["config-files/skull1.json"]
    views = [8, 16, 32]
    imgres = [256]
    tfmodes = ["texture"]
    tfres = [64]
    optimizers = ["Adam", "Adadelta"]
    lrs = [0.8]
    image_losses = [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1),
                    (1,1,0,0), (1,1,2,0), (1,1,0,5), (1,1,1,5)] # l1, l2, dssim, lpips
    prior_losses = [0.0, 0.01]

    iterations = 200
    batch_size = 4

    script_dir = os.path.abspath(script_dir)
    print("Write scripts to", script_dir)
    os.makedirs(script_dir, exist_ok=True)

    for settings, v, r, tf, R, o, lr, il, pl in itertools.product(
            settings_files, views, imgres, tfmodes, tfres, optimizers, lrs, image_losses, prior_losses):
        basename = os.path.splitext(os.path.split(settings)[1])[0]
        filename = "%s-v%d-r%d-tf%s-R%d-o%s-lr%04d-loss%d%d%d%d-%04d" % (
            basename, v, r, tf, R, o, int(lr*1000), il[0], il[1], il[2], il[3], int(pl*1000))
        output_file = os.path.join(output_dir, filename+".hdf5")
        loss_norm = 1 / (il[0]+il[1]+il[2]+il[3])
        args = "%s %s -v %d -r %d -tf %s -R %d -o %s -lr %.5f -i %d -b %d -l1 %.5f -l2 %.5f -dssim %.5f -lpips %.5f -ps %.5f" % (
            output_file, settings, v, r, tf, R, o, lr, iterations, batch_size,
            il[0]*loss_norm, il[1]*loss_norm, il[2]*loss_norm, il[3]*loss_norm, pl)
        with open(os.path.join(script_dir, filename+".sh"), "w") as f:
            f.write(launcher_file + " " + args + "\n")

if __name__ == '__main__':
    create_files(
        "results/tf/meta",
        "runs/open",
        "python tests/tf/train_meta.py")