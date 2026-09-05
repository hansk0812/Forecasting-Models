from data_provider import Dataset_Weather
import numpy as np

import matplotlib
from matplotlib import colors as mcolors
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt

from dataclasses import dataclass, field, asdict

#TODO Change typing per argument
# dataclasses needs type hinting to convert object to dict
from typing import Any

import argparse

@dataclass
class Config:
    
    #"""
    root_path: Any = None
    data_path: Any = None
    
    #[args.seq_len, args.label_len, args.pred_len],
    # size=(720, 0, 720)
    
    # 'S', "SM", 'M'
    features: Any = 'M'
            
    # Only for features='S'
    target: Any = None
    
    # Add datetime metadata alongwith data
    timeenc: Any = True
    
    # Time interval per timestep
    freq: Any = "15min"
    
    # For CycleNet's cycling over batches w.r.t. dataset timestamps
    cycle: Any = 32
    
    # Normalization: zscore, instance, None
    scale: Any = "instance"
    
    # Weather5K dataset: Hourly or None
    #seasonal_patterns: Any = None

    # Select first N variates if integer
    select_variates: Any = None

if __name__ == "__main__":

    config = Config()

    ap = argparse.ArgumentParser()
    ap.add_argument("root_path", help="Directory of dataset file")
    ap.add_argument("data_path", help="Dataset file in root_path")
    args = ap.parse_args()
    
    config.root_path = args.root_path
    config.data_path = args.data_path

    config = asdict(config)
    config["size"] = (720, 0, 720)

    subsets = None
    
    subsets = {"train": (None, None), "val": (None, None), "test": (None, None)}
    subset_flag = False
    
    for split in ["train", "val", "test"]:

        #if split == "train":
        colors_and_positions = [
            (0.00, '#0d233a'), (0.04236, '#70a1ff'),
            (0.04236, '#004d40'), (0.08472, '#80cbd3'),
            (0.08472, '#8c5000'), (1.00, '#ffda66')
            ]
        
        cmap = mcolors.LinearSegmentedColormap.from_list("flat_region_cmap", colors_and_positions)

        config["flag"] = split

        dataset = Dataset_Weather(**config)

        # Dataset variable placeholder: data_x and data_x2 for Dataset_Weather
        time_series = dataset.data_x
        if args.data_path == '2':
            time_series = np.concatenate((time_series, dataset.data_x2), axis=0)

        cov = np.cov(time_series.T)

        # Use HSV colormap instead 
        # Represent the distribution more
#        min_, max_ = cov.min(), cov.max()
#        factor = 0.4
#        cov[cov > factor * (max_ - min_)] = factor * (max_ - min_)
        
        if not subsets is None:
            beg, end = subsets[split]
            min_, max_ = cov.min(), cov.max()
            if beg is None:
                beg = min_
            if end is None:
                end = max_

            cov[cov < beg] = np.nan
            cov[cov > end] = np.nan

        cov = np.tril(cov)
        cov[cov==0] = np.nan
        
        fig, ax = plt.subplots()
        img = ax.imshow(cov, cmap=cmap, vmin=np.nanmin(cov), vmax=np.nanmax(cov))
        #img = ax.imshow(cov, cmap="gist_rainbow", vmin=np.nanmin(cov), vmax=np.nanmax(cov))
        plt.colorbar(img, ax=ax)
        
        ax.set_xticklabels([""] + dataset.vars, minor=False, rotation=45, ha="right")
        ax.set_yticklabels([""] + dataset.vars, minor=False)
        
        rows, cols = cov.shape
        for i in range(rows):
            for j in range(cols):
                val = cov[i, j]
                
                if not np.isnan(val):
                    t = ax.text(j, i, "%d" % int(val), 
                                ha="center", va="center", 
                                color="white", fontsize=6)
                    t.set_bbox(dict(facecolor="black", alpha=0.25, linewidth=0))

        if subset_flag:
            ax.set_title("%s %s set: %s" % ("JenaWeather (%d to %d) / (%d to %d)" % 
                                            (min_ if beg > min_ else beg, 
                                             max_ if end > max_ else end, 
                                             min_, max_), split, "Max-Planck-Institute"))
        else:
            ax.set_title("%s %s set: %s" % ("JenaWeather", split, "Max-Planck-Institute"))

        plt.savefig("%s_%s-set_covariance.pdf" % ("JenaWeather", split), dpi=300, bbox_inches="tight")
