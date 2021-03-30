"""
Creates the script files for train_meta.py
"""

import os
import itertools

def create_files(output_dir, script_dir, reference_file, basename, launcher_file):
    settings_files = ["config-files/skull2.json"]

    cropsizes = [64, 128]
    cropnumbers = [4,8]
    views = [4]
    prior_losses = [0.0, 0.01, 0.05]
    style_layers = [
        "conv_1:1,conv_3:1,conv_5:1",
        "conv_1:1,conv_3:1,conv_5:5",
        "conv_1:5,conv_3:1,conv_5:1",
        "conv_5:1,conv_7:1,conv_9:1",
    ]
    seeds = [42, 43, 44]

    imgres = 512
    tfres = 64
    tfmode = "texture"
    optimizer = "Adam"
    lr = 0.8
    iterations = 200
    minDensity = 0.075

    batch_size = 2

    script_dir = os.path.abspath(script_dir)
    print("Write scripts to", script_dir)
    os.makedirs(script_dir, exist_ok=True)

    for settings, cs, cn, v, ps, sl, seed in itertools.product(
        settings_files, cropsizes, cropnumbers, views, prior_losses, style_layers, seeds):
        filename = "%s-cs%d-cn%d-v%d-ps%04d-%s-s%03d" % (
            basename, cs, cn, v, int(ps*1000), sl.replace(':','x'), seed)
        output_file = os.path.join(output_dir, filename+".hdf5")
        args = "%s %s %s -cs %d -cn %d -v %d -ps %f --styleLayers %s --seed %d -r %d -tf %s -R %d --minDensity %f -o %s -lr %f -i %d -b %d" % (
            output_file, settings, reference_file,
            cs, cn, v, ps, sl, seed,
            imgres, tfmode, tfres, minDensity, optimizer, lr, iterations, batch_size)
        with open(os.path.join(script_dir, filename+".sh"), "w") as f:
            f.write(launcher_file + " " + args + "\n")

if __name__ == '__main__':
    references = [
        ("reference-body", "skull-refBody"),
        ("reference-skull", "skull-refSkull"),
        ("reference-thorax", "skull-refThorax"),
        ("reference-jet", "skull-refJet")
    ]
    for ref_img, basename in references:
        create_files(
            "results/tf/style3",
            "runs/open",
            "config-files/" + ref_img + ".png",
            basename,
            "python tests/tf/train_styletransfer.py")