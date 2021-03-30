"""
Creates the script files for train_meta.py
"""

import os
import itertools

def create_files(output_dir, script_dir, basename, launcher_file):
    settings_files = ["config-files/skull2.json"]

    views = [8, 32]
    #image_losses = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
    #                (1, 1, 0, 0), (1, 1, 2, 0), (1, 1, 0, 5), (1, 1, 1, 5)]  # l1, l2, dssim, lpips
    image_losses = [(1, 0, 0, 0), (0, 1, 0, 0)]  # l1, l2, dssim, lpips
    prior_losses = [0.0, 0.01, 0.1]
    minOpacity = [0.0, 0.1, 0.5]
    onlyOpacityUntil = [0, 50, 100]
    seeds = [42]

    tfmode = "texture"
    lr = 0.8
    iterations = 200
    batch_size = 8
    volumeSaveFrequency = 10

    script_dir = os.path.abspath(script_dir)
    print("Write scripts to", script_dir)
    os.makedirs(script_dir, exist_ok=True)

    for settings, v, il, pl, mo, oou, seed in itertools.product(
        settings_files, views, image_losses, prior_losses, minOpacity, onlyOpacityUntil, seeds):
        filename = "%s-tf%s-v%d-loss%d%d%d%d-%04d-mo%02d-oou%03d-s%03d" % (
            basename, tfmode, v, il[0], il[1], il[2], il[3], int(pl*1000), int(mo*10), int(oou*100), seed)
        output_file = os.path.join(output_dir, filename+".hdf5")
        loss_norm = 1 / (il[0] + il[1] + il[2] + il[3])
        args = "%s %s -v %d -tf %s -l1 %.5f -l2 %.5f -dssim %.5f -lpips %.5f -ps %.5f --minOpacity %f --onlyOpacityUntil %d -i %d --volumeSaveFrequency %d -b %d" % (
            output_file, settings, v, tfmode,
            il[0] * loss_norm, il[1] * loss_norm, il[2] * loss_norm, il[3] * loss_norm, pl,
            mo, oou, iterations, volumeSaveFrequency, batch_size
        )
        with open(os.path.join(script_dir, filename+".sh"), "w") as f:
            f.write(launcher_file + " " + args + "\n")

def createFiles2_Density(settings_file, output_prefix):
    # python ./tests/volume/train_volume.py ./results/volume/skull6big/identity-r8r256-ps5-l1.hdf5 ./config-files/skull5identity.json
    # -v 256 -tf texture -R 256 --multiscale 8 -I gauss -l1 1 -ps 5.0 -bm stochastic -lr 0.05 -i 200 --volumeSaveFrequency 20
    output_file = "results/volume/big3/" + output_prefix + "-density.hdf5"
    script_file = "runs/open/" + output_prefix + "-density.sh"
    with open(script_file, "w") as f:
        f.write("python ./tests/volume/train_volume.py %s %s -v 256 -tf texture -R 256 --multiscale 8 -I gauss -l1 1 -ps 5.0 -bm stochastic -lr 0.05 -i 200 --volumeSaveFrequency 20"%(
            output_file, settings_file))

def createFiles2_Color(settings_file, output_prefix):
    # python ./tests/volume/train_volume.py ./results/volume/skull6big/identity-r8r256-ps5-l1.hdf5 ./config-files/skull5identity.json
    # -v 256 -tf texture -R 256 --multiscale 8 -I gauss -l1 1 -ps 5.0 -bm stochastic -lr 0.05 -i 200 --volumeSaveFrequency 20
    output_file = "results/volume/big3/" + output_prefix + "-direct.hdf5"
    script_file = "runs/open/" + output_prefix + "-direct.sh"
    with open(script_file, "w") as f:
        f.write("python ./tests/volume/train_volume.py %s %s -v 256 -tf gauss -R 256 --multiscale 8 -I gauss -l1 1 -ps 0.1 -bm stochastic -lr 0.05 -i 200 --volumeSaveFrequency 20"%(
            output_file, settings_file))

    output_file = "results/volume/big3/" + output_prefix + "-pre.hdf5"
    script_file = "runs/open/" + output_prefix + "-pre.sh"
    with open(script_file, "w") as f:
        f.write(
            "python ./tests/volume/train_volume.py %s %s --preshaded -v 256 -tf gauss -R 256 --multiscale 8 -I gauss -l1 1 -ps 0.1 -bm stochastic -lr 0.05 -i 200 --volumeSaveFrequency 20" % (
                output_file, settings_file))

if __name__ == '__main__':
    #create_files(
    #    "results/volume/recon2",
    #    "runs/open",
    #    "skullrecon",
    #    "python tests/volume/train_volume.py")

    createFiles2_Density("./config-files/tooth3linear.json", "tooth3linear")
    createFiles2_Density("./config-files/thorax1linear.json", "thorax1linear")

    createFiles2_Color("./config-files/tooth3gauss.json", "tooth3gauss")
    createFiles2_Color("./config-files/thorax1gauss.json", "thorax1gauss")