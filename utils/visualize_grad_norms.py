from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from collections import OrderedDict
import matplotlib.colors as mcolors

import argparse

import os
import glob
import numpy as np

import random

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="logs/gradnorms folder", default=None)
    ap.add_argument("--models", nargs='+', help="Models to include in visualization. Ignore for all available models.", default=None)
    args = ap.parse_args()

    H = [96, 192, 336, 720]
    
    fig, ax = plt.subplots()
    plots = OrderedDict()
    plot_colors = list(mcolors._colors_full_map.values())
    random.shuffle(plot_colors)
    
    # midpoints
    midpts, midpts_plot = {}, {}

    for h in H:

        for cutoff_type in ["forward", "backward"]:

            fnames = sorted(glob.glob("%s/*_%d_%s_gradnorms.txt" % (args.folder, h, cutoff_type)))
        
            if len(fnames) == 0:
                print ("H=%d files not found!" % h); exit()
            
            for idx, fname in enumerate(fnames):
                model = fname.split('_')[0].split('/')[-1]
                
                if not args.models is None and not model in args.models:
                    continue

                if not model in plots:
                    plots[model] = []
                    midpts[model] = []
                with open(fname, 'r') as f:
                    values = []
                    for line in f.readlines():
                        if not "Grad" in line:
                            continue
                        values.append(float(line.split(": ")[-1]))
                
                p, = plt.plot(np.arange(0, len(values)), values, label=model, 
                                color=plot_colors[idx])
                if len(midpts[model]) == 0:
                    midpts[model].append(np.array(values))
                else:
                    midpts[model] = np.argwhere(np.diff(np.sign(np.array(values)-midpts[model])))
                plots[model].append(p)
            
            plt.legend([tuple(plots[model]) for model in plots], list(plots.keys()),
                                        handler_map={tuple: HandlerTuple(ndivide=None)})
        #plt.show()
        #plt.savefig("gradnorms_%d_%s.png" % (h, "all_models" if args.models is None else "_".join(sorted(args.models))))
        
        # midpoints: 96, 192, 336, 720
        for k in midpts:
            if not k in midpts_plot:
                midpts_plot[k] = [midpts[k][0][1]]
            else:
                midpts_plot[k].append(midpts[k][0][1])
        
        for k in plots:
            midpts[k] = []
            plots[k] = []

    plt.clf()

    label_plots = []
    for idx, k in enumerate(midpts_plot):
        p, = plt.plot(H, midpts_plot[k], label=k, marker='o', color=plot_colors[idx])
        label_plots.append(p)

    for h in H:
        p, = plt.plot([3*h/4, 5*h/4], [h/2, h/2], linestyle="--", color='black', label="H/2 Line")
        label_plots.append(p)
    
    label_keys = list(midpts_plot.keys())
    label_keys.append("H/2 Line")
    plt.legend(label_plots, label_keys, handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.title("At horizon=y, the first y predictions' gradient updates' norms are the same as those of the last H-y!")
    
    plt.xticks(H, ["H=%d" % H[idx] for idx in range(len(H))])

    plt.show()

