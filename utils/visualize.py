from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from collections import OrderedDict
import matplotlib.colors as mcolors

import argparse

import os
import glob
import re

import numpy as np

import random

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", help="[gradnorms, autocorr=horizon/nlags]")
    ap.add_argument("folder", help="logs/gradnorms or logs/autocorr folder", default=None)
    ap.add_argument("--models", nargs='+', help="Models to include in visualization. Ignore for all available models.", default=None)
    args = ap.parse_args()

    assert args.mode == "gradnorms" or "autocorr" in args.mode

    H = [96, 192, 336, 720]
    
    fig, ax = plt.subplots()

    plots = OrderedDict()
    plot_colors = list(mcolors._colors_full_map.values())
    random.shuffle(plot_colors)
    
    # midpoints
    midpts, midpts_plot = {}, {}

    for h_idx, h in enumerate(H):
        
        if "autocorr" in args.mode:
            autocorrs_gt = []

        types = ["forward", "backward"] if args.mode == "gradnorms" else [str(int(h/int(args.mode.split('=')[-1])))]
        for cutoff_type in types:
            
            fnames = sorted(glob.glob(os.path.join(args.folder, "*_%d_%s_%s.txt" % (
                h, cutoff_type, args.mode.split('=')[0]))))
            
            for idx in range(len(fnames)-1,-1,-1):
                if not args.models is None and not fnames[idx].split('_')[0].split('/')[-1] in args.models:
                    del fnames[idx]
            
            if len(fnames) == 0:
                print ("H=%d files not found!" % h if args.mode == "gradnorms" else int(h/int(cutoff_type))); exit()
            
            for idx, fname in enumerate(fnames):
                model = fname.split('_')[0].split('/')[-1]
                
                if args.mode == "gradnorms":
                    if not model in plots:
                        plots[model] = []
                        midpts[model] = []
                    with open(fname, 'r') as f:
                        values = []
                        for line in f.readlines():
                            if not "Grad " in line:
                                continue
                            values.append(float(line.split(": ")[-1]))
                    
                    p, = plt.plot(np.arange(0, len(values)), values, label=model, 
                                    color=plot_colors[idx])
                    if len(midpts[model]) == 0:
                        midpts[model].append(np.array(values))
                    else:
                        midpts[model] = np.argwhere(np.diff(np.sign(np.array(values)-midpts[model][0])))
                    plots[model].append(p)
                
                else:

                    flag, autocorrs = False, []
                    with open(fname, 'r') as f:
                        model =  fname.split('_')[0].split('/')[-1]
                        
                        for line in f.readlines():
                            if "Autocorrelation for" in line or flag:
                                
                                if not flag or "gt" in line:
                                    arr = line.split('[')[-1].replace("        ", "").strip() + " "
                                else:
                                    arr += line.replace(']', "").strip() + " "
                                    if ']' in line:
                                        arr = re.sub(r"\s+", ' ', arr).strip()
                                        autocorr = [float(x) for x in arr.split(' ')]
                                        autocorrs.append(autocorr)
                                flag = True
                    
                    plt.plot(list(range(len(autocorrs[0]))), autocorrs[0], label=model, color=plot_colors[idx+1])
                    autocorrs_gt.append(autocorrs[1])

            if "autocorr" in args.mode:
                autocorrs_gt = np.array(autocorrs_gt).mean(axis=0)
                plt.plot(list(range(len(autocorrs_gt))), autocorrs_gt, label="Ground Truth", color="black", linestyle="dashdot")
                plt.legend()
                plt.xlabel("H/4 lags")
                plt.ylabel("Autocorrelation")
                plt.title("Comparing ACF of models' test set predictions vs ground truth averages")
                #plt.show()
                plt.savefig("autocorrs_%d_%s.png" % (h, "all_models" if args.models is None else "_".join(sorted(args.models))))
                plt.clf()

            if args.mode == "gradnorms":
                plt.legend([tuple(plots[model]) for model in plots], list(plots.keys()),
                                            handler_map={tuple: HandlerTuple(ndivide=None)})
        if args.mode == "gradnorms":
            plt.show()
            plt.savefig("gradnorms_%d_%s.png" % (h, "all_models" if args.models is None else "_".join(sorted(args.models))))
        
            # midpoints: 96, 192, 336, 720
            for k in midpts:
                if not k in midpts_plot:
                    midpts_plot[k] = [midpts[k][0][0]]
                else:
                    midpts_plot[k].append(midpts[k][0][0])
            
            for k in plots:
                midpts[k] = []
                plots[k] = []
    
    if args.mode != "gradnorms":
        exit()

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

