import matplotlib.pyplot as plt
import numpy as np
import os

def plot_voxels(voxels, save_dir, filename):
    # Make sure voxels are *not* batched
    assert len(voxels.shape) == 3

    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels, edgecolor='k')

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()