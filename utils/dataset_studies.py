from data_provider import Dataset_Weather
import numpy as np
import torch

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

def temporal_covariance(dataset, title, ticks, cmap, stride=None, save_file=None, bounds=None):
    
    V = dataset.get_num_features()

    device = torch.device("cuda")
    for idx in range(V):

        full_dataset = []
        for sample_idx, data in enumerate(dataset):
            if not stride is None:
                if sample_idx % stride != 0:
                    continue
                else:
                    input_window = data[0][:, idx]
            else:
                input_window = (data[0][:, idx] * 10**5).astype(np.int16)

            full_dataset.append(input_window)

        full_dataset = np.stack(full_dataset)
        full_dataset = full_dataset[~np.isnan(full_dataset).any(axis=1)]
        
        full_dataset = torch.tensor(full_dataset.T).to(device)
        cov = torch.cov(full_dataset)

        cov = cov.cpu().numpy()
        
        if not bounds is None:
            bounds_var = [[cov.min() + x[0] * (cov.max() - cov.min()), x[1]] for x in bounds]

        save_file_extn = save_file.split('.')[-1]
        save_file_var = '.'.join(save_file.split('.')[:-1]) + "V%d" % idx + '.' + save_file_extn
        plot_covariance_heatmap(cov, title, ticks, cmap, save_file_var, bounds_var)

def plot_covariance_heatmap(cov, title, ticks=None, cmap="inferno", save_file=None, bounds=None):
    
    # Use bounds for rescaling colorbar w.r.t. how small they might be 

    cov = np.tril(cov)
    cov[cov==0] = np.nan

    fig, ax = plt.subplots()
    img = ax.imshow(cov, cmap=cmap, vmin=np.nanmin(cov), vmax=np.nanmax(cov))

    if not bounds is None:
        cmap = mcolors.ListedColormap([x[1] for x in bounds[:-1]])
        norm = mcolors.BoundaryNorm(boundaries=[x[0] for x in bounds], ncolors=cmap.N)
        plt.colorbar(img, ax=ax, norm=norm, boundaries=[x[0] for x in bounds])
    else:
        plt.colorbar(img, ax=ax)
    
    if not ticks is None:
        ax.set_xticklabels([""] + ticks, minor=False, rotation=45, ha="right")
        ax.set_yticklabels([""] + ticks, minor=False)
    
    rows, cols = cov.shape
    if rows < 50:
        for i in range(rows):
            for j in range(cols):
                val = cov[i, j]
                
                if not np.isnan(val):
                    t = ax.text(j, i, "%d" % int(val), 
                                ha="center", va="center", 
                                color="white", fontsize=6)
                    t.set_bbox(dict(facecolor="black", alpha=0.25, linewidth=0))

    ax.set_title(title)
    
    if not save_file is None:
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":

    config = Config()

    ap = argparse.ArgumentParser()
    ap.add_argument("root_path", help="Directory of dataset file")
    ap.add_argument("data_path", help="Dataset file in root_path")
    ap.add_argument("--per_variate_covariance", action="store_true", help="If True, doesn't calculate per timestep covariance")
    args = ap.parse_args()
    
    config.root_path = args.root_path
    config.data_path = args.data_path

    config = asdict(config)
    config["size"] = (720, 0, 720)

    subsets = None
    
    subsets = {"train": (None, None), "val": (None, None), "test": (None, None)}
    subset_flag = False
    
    for split in ["train", "val", "test"]:

        config["flag"] = split

        dataset = Dataset_Weather(**config)

        # Dataset variable placeholder: data_x and data_x2 for Dataset_Weather
        time_series = dataset.data_x
        
        # Heatmap title
        title_str = "Beutenberg"
        if subset_flag:
            title = "%s %s set: %s" % ("JenaWeather (%d to %d) / (%d to %d)" % 
                                            (min_ if beg > min_ else beg, 
                                             max_ if end > max_ else end, 
                                             min_, max_), split, title_str)
        else:
            title = "%s %s set: %s" % ("JenaWeather", split, title_str)
       
        if args.data_path == '2':
            time_series = np.concatenate((time_series, dataset.data_x2), axis=0)
            title += "; Saaleaue"

        if args.per_variate_covariance:
            colors_and_positions = [
                (0.00, '#0d233a'), (0.0236, '#70a1ff'),
                (0.0236, '#004d40'), (0.0472, '#80cbd3'),
                (0.0472, '#8c5000'), (1.00, '#ffda66')
                ]
            cmap = mcolors.LinearSegmentedColormap.from_list("flat_region_cmap", colors_and_positions)

        else:
            colors_and_positions = [
                (0.00, '#0d233a'), (0.0025, '#70a1ff'),
                (0.0025, '#004d40'), (0.005, '#80cbd3'),
                (0.0075, '#8c5000'), (1.00, '#ffda66')
                ]
            cmap = mcolors.LinearSegmentedColormap.from_list("flat_region_cmap", colors_and_positions)
            bounds = [colors_and_positions[idx] for idx in range(0, len(colors_and_positions), 2)] + [(1, "")]

            colors_and_positions_strided = [
                    (0.00, "#3321b3"), (0.00005, "#05023d"),
                    (0.00005, "#9313db"), (0.0001, "#3c0061"),
                    (0.0001, "#d64ca7"), (0.00015, "#6b1254"),
                    (0.00015, "#ff7b79"), (0.9995, "#9c3533"),
                    (0.9995, "#ffbe5c"), (0.99975, "#b86d00"),
                    (0.99975, "#faff66"), (1, "#a3ab00"),
                ]

            cmap_strided = mcolors.LinearSegmentedColormap.from_list("flat_region_cmap", colors_and_positions_strided)
            bounds_strided = [colors_and_positions_strided[idx] for idx in range(0, len(colors_and_positions_strided), 2)] + [(1, "")]

        if not args.per_variate_covariance:
            #save_file = "%s_%s-set_temporal_covariance.pdf" % ("JenaWeather_" + title_str.replace("; ", '_'), split)
            #temporal_covariance(dataset, title, None, cmap, stride=None, save_file=save_file, bounds=bounds)
        
            save_file = "%s_%s-set_strided_temporal_covariance.pdf" % ("JenaWeather_" + title_str.replace("; ", '_'), split)
            temporal_covariance(dataset, title, None, cmap_strided, stride=config["size"][-1], save_file=save_file, bounds=bounds_strided)

        # Skip per-variate covariance plots
        continue

        cov = np.cov(time_series.T)

        # Use HSV colormap instead of below 
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

        #cov = np.tril(cov)
        #cov[cov==0] = np.nan
        
        save_file = "%s_%s-set_covariance.pdf" % ("JenaWeather_" + title_str.replace("; ", '_'), split)
        ticks = dataset.vars

        plot_covariance_heatmap(cov, title, ticks, cmap, save_file=save_file)

        """
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
                                             min_, max_), split, title_str))
        else:
            ax.set_title("%s %s set: %s" % ("JenaWeather", split, title_str))

        plt.savefig("%s_%s-set_covariance.pdf" % ("JenaWeather_" + title_str.replace("; ", '_'), split), dpi=300, bbox_inches="tight")
        """
