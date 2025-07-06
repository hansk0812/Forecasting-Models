from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from collections import OrderedDict
import matplotlib.colors as mcolors

import argparse

import os
import glob
import numpy as np
import argparse

if __name__ == "__main__":
    
    #ap = argparse.ArgumentParser()
    #ap.add_argument("mode", help="[horizon, model]")
    #args = ap.parse_args()
    
    models = []
    plots = OrderedDict()
    plot_colors = list(mcolors._colors_full_map.values())
    print (len(plot_colors))
    for h in [96, 192, 336, 720]:

        for cutoff_type in ["forward", "backward"]:

            fnames = glob.glob("logs/gradnorms/*_%d_%s_gradnorms.txt" % (h, cutoff_type))
        
            if len(fnames) == 0:
                print ("H=%d files not found!" % h); exit()
            
            for idx, fname in enumerate(fnames):
                model = fname.split('_')[0].split('/')[-1]
                if not model in plots:
                    plots[model] = []
                with open(fname, 'r') as f:
                    values = []
                    for line in f.readlines():
                        if "Batch" in line:
                            continue
                        values.append(float(line.split(": ")[-1]))
                
                p, = plt.plot(np.arange(0, h), values, label=model, color=plot_colors[idx])
                plots[model].append(p)

            #if cutoff_type == "backward":
            #    plt.title("Mean of gradient norms of the model where the first x timesteps are zero for H=%d models" % h)
            #else:
            #    plt.title("Mean of gradient norms of the model over the first x timesteps for H=%d models" % h)

            #plt.legend()
            #plt.show()
        
        plt.legend([tuple(plots[model]) for model in plots], list(plots.keys()),
               handler_map={tuple: HandlerTuple(ndivide=None)})
        plt.show()
        
