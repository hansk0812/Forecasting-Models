import argparse
import os, glob

import numpy as np

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("tkAgg")

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Directory with saved npy files for std calculations")
    ap.add_argument("dataset", help="Name of dataset")
    ap.add_argument("model", help="Name of model")
    ap.add_argument("horizon_size", help="Size of forecast horizon", type=int)
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.folder, "%s_%s_%d_*.npy" % (args.dataset, args.model, args.horizon_size)))

    assert len(files) >= 5, "At least 5 runs needed for visualization"
    
    error_vals = []
    for fl in files:
        with open(fl, "rb") as f:
            err = np.load(f)
            error_vals.append(err)

    error_vals = np.stack(error_vals)
    
    error_means = error_vals.mean(axis=0)
    error_stds = error_vals.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    
    start = 5
    heatmap_means = error_means.copy()
    heatmap_means[:start] = error_means.min()

    x = np.arange(1, args.horizon_size + 1, 1)
    extent = [x[0] - (x[1] - x[0]) / 2., x[-1] + (x[1] - x[0]) / 2., 0, 1]
 
    ax1.imshow(heatmap_means[np.newaxis, :], cmap="inferno", aspect="auto", extent=extent)
    ax1.set_yticks([])
    ax1.set_xlim(extent[0], extent[1])
   
    ax2.plot(range(1, args.horizon_size + 1), error_means, label="MSE Per Timestep")
    ax2.fill_between(range(1, args.horizon_size + 1), error_means - error_stds, error_means + error_stds, 
                     color='grey', alpha=0.7)
    
    ax2.text(0.6, 0.05, "STD ∈ [%.2e, %.2e]" % (error_stds.min(), error_stds.max()), 
             transform=ax2.transAxes, fontsize=10)
    ax2.legend(fontsize=10)
    
    plt.savefig(os.path.join(args.folder, "%s_%s_%d_heatmap.pdf" % (args.dataset, args.model, args.horizon_size)),
                dpi=300, bbox_inches="tight")
