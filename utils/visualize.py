from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from collections import OrderedDict

import argparse

import os
import glob
import re

import numpy as np

import random

import warnings
warnings.filterwarnings("ignore")

# https://www.github.com/hanskrupakar/COCO-Style-Dataset-Generator-GUI 
# (Trapezoid formula based shoelace (Gauss') formula)
def find_poly_area(coords, h):
    # find the area under the curve using its curve polygon 
    # against the ideal case of gradient norm distributions
    # across the horizon timesteps from 0 to h
    # the area is negative if the curve is below the H=0 --> H=H line as convention
    
    index = np.where(coords[:,0] == h)[0][0]
    
    poly = coords[:index+1,:]
    
    poly = np.concatenate((poly, np.array([[
                index,
                (((coords[-1][1]-coords[0][1]))/float(coords[-1][0]-coords[0][0]))*h + float(coords[0][1])
                ]])), axis=0)
    
    line_x = np.array(list(range(index+1)))
    line_y = ((coords[-1][1]-coords[0][1])/float(coords[-1][0]-coords[0][0]))*line_x + coords[0][1]
    
    intersection_pts = np.argwhere(np.diff(np.sign(poly[:-1,1]-line_y))).flatten().tolist()

    if len(intersection_pts) > 0:
        if intersection_pts[-1] == index - 1:
            intersection_pts[-1] = index
        if len(intersection_pts) == 1 and intersection_pts[0] != 0:
            intersection_pts = [0] + intersection_pts
        if intersection_pts == [0]:
            intersection_pts.append(index)
        else:
            if intersection_pts[-1] != index:
                intersection_pts += [index]
    else:
        intersection_pts = [0, index]

    polys, poly_adds = [], []
    for idx in range(len((intersection_pts))-1):
        p = poly[intersection_pts[idx]:intersection_pts[idx+1]+1,:].tolist()
        if idx >= 1:
            start_pt = poly[intersection_pts[idx]][0].astype(np.int32)
            p = [[line_x[start_pt], line_y[start_pt]]] + p + \
                    [[line_x[min(len(line_x)-1, intersection_pts[idx+1])], line_y[min(len(line_y)-1, intersection_pts[idx+1])]]]
            poly_adds.append(2)
        else:
            if p[-1][1] != line_y[-1]: 
                # eliminate line overlap polygon points as last points
                if len(intersection_pts) > 2:
                    p = p + [[line_x[intersection_pts[idx+1]], line_y[intersection_pts[idx+1]]]]
                else:
                    p = p + [[line_x[-1], line_y[-1]]]
                poly_adds.append(1)
            else:
                poly_adds.append(0)
        polys.append(np.array(p))
    
    signs = []
    for idx, p in enumerate(polys):
        if idx == 0 or len(signs) == 0:
            if line_y[p.shape[0]//2] != p[p.shape[0]//2,1]:
                signs.append(line_y[p.shape[0]//2] < p[p.shape[0]//2,1])
        else:
            signs.append(not signs[-1])

    return_area = 0
    for p, sign in zip(polys, signs):
        
        # coords: np.array([[x_i,y_i],...])
        x, y = p[:,0], p[:,1]
        return_area += (2*sign-1)*(0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))))/2 #shoelace algorithm
    
    return return_area

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", help="[gradnorms, autocorr=horizon/nlags]", default="autocorr=1")
    ap.add_argument("folder", help="logs/gradnorms or logs/autocorr folder", default=None)
    ap.add_argument("--models", nargs='+', help="Models to include in visualization. Ignore for all available models.", default=None)
    ap.add_argument("--start_color_idx", default=[0], nargs='+', help="Color idx for single color per model", type=int)
    args = ap.parse_args()

    assert args.mode == "gradnorms" or "autocorr" in args.mode

    H = [96, 192, 336, 720]
    
    plots = OrderedDict()
    
    # select a few colors
    #import matplotlib.colors as mcolors
    #plot_colors = list(mcolors._colors_full_map.values())
    #random.shuffle(plot_colors)
    
    plot_colors = ["#56ae57", "#894585", "#a5a391", "#0c06f7", "#61de2a", "#ff0789", "#d3b683", \
                   "#430541", "#d0e429", "#fdb147", "#850e04", "#efc0fe", "#8fae22", "goldenrod", "crimson"]
    plot_colors_per_model = np.array(plot_colors)[args.start_color_idx]
    
    # heatmap for CycleNet epochs curves
    #import matplotlib as mpl
    #cmap = mpl.colormaps['cividis']
    # Take colors at regular intervals spanning the colormap.
    #colors = cmap(np.linspace(0.2, 1, 6))   
    #plot_colors_per_model = np.array(colors)[args.start_color_idx]

    for h in H:

        fnames_forward = sorted(glob.glob(os.path.join(args.folder, "*_%d_%s_%s.txt" % (
            h, "forward", args.mode))))
        fnames_backward = [x.replace("forward", "backward") for x in fnames_forward]

        for f_f, f_b in zip(fnames_forward, fnames_backward):

            if not args.models is None and not any([x in f_f for x in args.models]):
                continue
            
            ctx = 0
            lines_f, lines_b = [], []
            with open(f_f, 'r') as f:
                for line in f.readlines():
                    if "Grad " in line:
                        ctx += 1
                        lines_f.append(line)
                if ctx > h:
                    continue
            ctx = 0
            with open(f_b, 'r') as f:
                for line in f.readlines():
                    if "Grad " in line:
                        ctx += 1
                        lines_b.append(line)
                if ctx > h:
                    continue
            
            if len(lines_f) == 0 or len(lines_b) == 0:
                print ("Missing file:", f_f, f_b)
                continue

            #with open(f_f, 'w') as f:
            #    f.writelines(lines_f)
            #with open(f_b, 'w') as f:
            #    f.writelines(lines_b)
    
    types = ["forward", "backward"] if args.mode == "gradnorms" else [str(h) for h in H]
    loss_based_weights = {h: {c: [] for c in types} for h in H}

    # midpoints
    midpts, midpts_plot = {}, {}
    areas = []
    for h_idx, h in enumerate(H):
        
        fig, ax = plt.subplots()
    
        if "autocorr" in args.mode:
            autocorrs_gt = []
        else:
            poly_areas = OrderedDict({k: {} for k in types})
        
        plot_diffs = {}
        for cutoff_type in types:
            
            #if cutoff_type == "forward":
            #    continue

            fnames = sorted(glob.glob(os.path.join(args.folder, "*_%d_%s_%s.txt" % (
                h, cutoff_type, args.mode.split('=')[0]))))

            for idx in range(len(fnames)-1,-1,-1):
                if not args.models is None and not fnames[idx].split('/')[-1].split('_')[0] in args.models:
                    del fnames[idx]
            
            if len(fnames) == 0:
                print ("H=%d files not found!" % h if args.mode == "gradnorms" else int(h/int(cutoff_type))); continue #exit()
            
            for idx, fname in enumerate(fnames):
                model = fname.split('/')[-1].split('_')[0]
                
                if not model in plot_diffs:
                    plot_diffs[model] = {}

                if args.mode == "gradnorms":

                    if not model in plots:
                        plots[model] = []
                        midpts[model] = []
                    with open(fname, 'r') as f:
                        values = []
                        for line in f.readlines():
                            if model.lower() == "spacetime" and "Gra " in line:
                                pt = float(line.split(": ")[-1])
                                loss_based_weights[h][cutoff_type].append(pt)
                            
                            if not "Grad " in line:
                                continue
                            values.append(float(line.split(": ")[-1]))
                    
                    p, = plt.plot(np.arange(0, len(values)), values, label=model, 
                                    color=plot_colors_per_model[idx], linewidth=0.5) 
                    plt.plot([0, len(values)-1], [values[0], values[-1]], color=plot_colors_per_model[idx], linestyle='--', linewidth=0.5) 
                    
                    if model == "SpaceTime":
                        if cutoff_type == "forward":
                            text = "H=%s: %.5f\n" % ("0  " if h < 100 else "0    ", loss_based_weights[h][cutoff_type][0])
                        else:
                            text =  text + "H=%d: %.5f" % (h, loss_based_weights[h][cutoff_type][0])
                        
                            plt.text(0.15, 0.5, text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                                    bbox={"facecolor": plot_colors_per_model[idx], "alpha": 0.5, "pad": 0.35, "boxstyle": "round"}, linespacing=1.5)

                    plot_diffs[model][cutoff_type] = values

                    #if model.lower() == "spacetime":
                    #    plt.plot(0, first_pt, marker='o', color=plot_colors_per_model[idx])
                    
                    # calculate area
                    poly_areas[cutoff_type][model] = []
                    pts = np.stack((np.arange(0, len(values)), values)).transpose()
                    
                    for h_ in range(1, h+1):
                        #plt.plot([0, len(values)-1], [values[0], values[-1]], color=plot_colors[idx], linestyle='--')
                        #plt.plot(list(range(h_)), values[:h_], color=plot_colors[idx], linestyle='--')
                        poly_areas[cutoff_type][model].append(find_poly_area(pts, h_))

                    if len(midpts[model]) == 0:
                        midpts[model].append(np.array(values))
                    else:
                        print ("Model: %s ; H = %d" % (model, h))
                        midpts[model] = np.argwhere(np.diff(np.sign(np.array(values)-midpts[model][0])))
                    plots[model].append(p)
                
                else:

                    flag, autocorrs = False, []
                    with open(fname, 'r') as f:
                        model =  fname.split('/')[-1].split('_')[0]
                        model_idx = args.models.index(model)
                        
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
                    
                    try:
                        plt.plot(list(range(len(autocorrs[0]))), autocorrs[0], label=model, color=plot_colors_per_model[model_idx])
                        autocorrs_gt.append(autocorrs[1])
                    except Exception:
                        import traceback
                        traceback.print_exc()

            if "autocorr" in args.mode:
                autocorrs_gt = np.array(autocorrs_gt).mean(axis=0)
                plt.plot(list(range(len(autocorrs_gt))), autocorrs_gt, label="Ground Truth", color="black", linestyle="dashdot")
                plt.legend()
                plt.xlabel("H=%d lags" % h)
                plt.ylabel("Autocorrelation")
                plt.title("Self attention models' ACF averages over the test set")
                plt.savefig("plots/autocorrs_%d_%s.pdf" % (h, "all_models" if args.models is None else "_".join(sorted(args.models))), dpi=600, bbox_inches="tight")
                #plt.show()
                plt.clf()

            if args.mode == "gradnorms":
                plt.legend([tuple(plots[model]) for model in plots], list(plots.keys()),
                        handler_map={tuple: HandlerTuple(ndivide=None)}, loc='center right', prop={"size": 6})

        if args.mode == "gradnorms":
#            ax_top = ax.twiny()
#            ax_top.set_xlim(ax.get_xlim())
#            ax.set_xlabel("Forward [0->%d]" % h)
#            ax_top.set_xlabel("Backward [%d->0]" % h)
#            reverse_ticks = list(reversed(ax.get_xticklabels()))
#            extent = int(reverse_ticks[0]._x - reverse_ticks[1]._x)
#            
#            if extent == 0:
#                continue
#            
#            start_idx = 0
#            while reverse_ticks[start_idx]._x > h:
#                start_idx += 1
#            if start_idx > 1:
#                start_idx -= 1
#            reverse_ticks[start_idx]._text = str(h)
#            reverse_ticks[start_idx]._x = h
#            for idx, text_obj in enumerate(reverse_ticks[start_idx + 1:]):
#                text_obj._text = str(int(reverse_ticks[start_idx + idx]._x)-extent)
#                text_obj._x = int(text_obj._text)
#            ax_top.set_xticklabels(reverse_ticks)
            plt.savefig("plots/gradnorms_%d_%s.pdf" % (h, "all_models" if args.models is None else "_".join(sorted(args.models))), dpi=600, bbox_inches="tight")
            #plt.show(); exit()
            
            plt.clf()
            min_y, midpts_diff = np.inf, {}
            for idx, model_name in enumerate(sorted(plot_diffs.keys())):
                diff = np.array(plot_diffs[model_name]["forward"]) - np.array(plot_diffs[model_name]["backward"])
                plt.plot(list(range(len(diff))), diff, label=model_name, color=plot_colors_per_model[idx])
                
                min_y = np.min(np.array([min_y, diff.min()]))
                midpts_diff[model_name] = np.abs(diff).argmin()
            
            for idx, model_name in enumerate(sorted(midpts_diff.keys())):
                plt.plot(midpts_diff[model_name], min_y, marker='o', markersize=3, color=plot_colors_per_model[idx])

            plt.legend(prop={"size": 6})
            plt.title("Difference between forward and backward mode gradient norm averages")
            plt.savefig("plots/gradnorms_%d_%s_diffs.pdf" % (h, "_".join(args.models)), dpi=600, bbox_inches="tight")
            plt.clf()
            
            """
            # Heatmaps slightly harder to interpret between models
            for idx, model_name in enumerate(sorted(plot_diffs.keys())):
                heatmap = np.zeros((h//2,h+1))
                for jdx in range(h//2):
                    diff = np.array(plot_diffs[model_name]["forward"])[:h-idx+1] - np.array(plot_diffs[model_name]["backward"])[idx:]
                    heatmap[jdx][idx:] = diff
                plt.imshow(heatmap, cmap='hot', interpolation='nearest')
                plt.title("%s Heatmap of Differences over the first h->h//2 values" % model_name)
                plt.show()
                plt.clf()
            plt.legend(prop={"size": 6})
            plt.title("Difference between forward and backward mode gradient norm averages")
            plt.savefig("gradnorms_%d_%s_diffs.pdf" % (h, "_".join(args.models)), dpi=600, bbox_inches="tight")
            plt.clf()
            """

            # midpoints: 96, 192, 336, 720
            for k in midpts:
                if not k in midpts_plot:
                    midpts_plot[k] = [midpts[k][0][0]]
                else:
                    midpts_plot[k].append(midpts[k][0][0])
            
            for k in plots:
                midpts[k] = []
                plots[k] = []
            
            fig, ax = plt.subplots()

            for idx, model in enumerate(poly_areas[cutoff_type].keys()):
                plt.plot([0, len(poly_areas[cutoff_type][model])], [0, 0], color="black")
                for cutoff_type in types:
                    plt.plot(np.arange(0, len(poly_areas[cutoff_type][model])),
                            poly_areas[cutoff_type][model], label=model + "[%s->%s]" % (
                                "0" if cutoff_type=="forward" else "x",
                                "x" if cutoff_type=="forward" else str(len(poly_areas[cutoff_type][model]))), 
                            color=plot_colors_per_model[idx],
                            linestyle="dashed" if cutoff_type==types[1] else "dotted")
    
            plt.legend(prop={"size": 6})
            
            #ax_top = ax.twiny()
            #ax_top.set_xlim(ax.get_xlim())
            #ax.set_xlabel("Forward [0->%d] (dotted line)" % h)
            #ax_top.set_xlabel("Backward [%d->0] (dashed line)" % h)
            #reverse_ticks = list(reversed(ax.get_xticklabels()))
            #extent = int(reverse_ticks[0]._x - reverse_ticks[1]._x)
            #start_idx = 0
            #while reverse_ticks[start_idx]._x > h:
            #    start_idx += 1
            #reverse_ticks[start_idx]._text = str(h)
            #reverse_ticks[start_idx]._x = h
            #for idx, text_obj in enumerate(reverse_ticks[start_idx + 1:]):
            #    text_obj._text = str(int(reverse_ticks[start_idx + idx]._x)-extent)
            #    text_obj._x = int(text_obj._text)
            #ax_top.set_xticklabels(reverse_ticks)
 
            plt.savefig("plots/gradnorms_%d_%s_areas.pdf" % (h, "all_models" if args.models is None else "_".join(sorted(args.models))), dpi=600, bbox_inches="tight")
            #plt.show()
            plt.clf()

            areas.append(poly_areas)
        #areas.append(poly_areas)

    if args.mode != "gradnorms":
        exit()

    plt.clf()

    label_plots = []
    for idx, k in enumerate(midpts_plot):
        p, = plt.plot(H, midpts_plot[k], label=k, marker='o', color=plot_colors_per_model[idx], linewidth=0.5) 
        label_plots.append(p)

    for h in H:
        p, = plt.plot([3*h/4, 5*h/4], [h/2, h/2], linestyle="--", color='black', label="H/2 Line")
        label_plots.append(p)
    
    label_keys = list(midpts_plot.keys())
    label_keys.append("H/2 Line")
    plt.legend(label_plots, label_keys, handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.title("Gradient equivariance points over the horizon")
    
    plt.xticks(H, ["H=%d" % H[idx] for idx in range(len(H))])
    
    plt.savefig("plots/gradnorms_%d_%s_midpts.pdf" % (h, "all_models" if args.models is None else "_".join(sorted(args.models))), dpi=600, bbox_inches="tight")
    #plt.show()
    
    fig, ax = plt.subplots()
    plt.plot([0, 1], [0, 0], color="black")

    # plot areas in a single 0-1 plot
    for idx, (h, area_dict) in enumerate(zip(H, areas)):
        maxs = {m: [] for m in area_dict["forward"].keys()}
        for k in area_dict:
            for m in area_dict[k]:
                area_dict[k][m] = np.array(area_dict[k][m])
                #mask = (area_dict[k][m]>0).astype(np.int32)
                #area_plot = np.sqrt(area_dict[k][m]*mask)
                #area_plot += -np.sqrt(-area_dict[k][m]*(1-mask))
                #area_plot /= np.sqrt(float(h))
                area_dict[k][m] = area_dict[k][m] / h
                maxs[m].append(max(area_dict[k][m].max(), (-area_dict[k][m]).max()))
        for k in area_dict:
            for m in area_dict[k]:
                l = "[ %s  ->  %s ]" % (
                        "0".rjust(9) if k=="forward" else ("x*%d"%h).rjust(6 if h>100 else 7), 
                        ("%d"%h).rjust(6 if h>100 else 7) if k=="backward" else ("x*%d"%h).rjust(5 if h>100 else 6))
                
                plt.plot(np.arange(0, 1, 1/len(area_dict[k][m])), 
                         area_dict[k][m]/max(maxs[m]), 
                         label=l, 
                         color=plot_colors[idx],
                         linestyle="dashed" if k=="backward" else "dotted")
            
    plt.legend(prop={"size": 6})
    #ax_top = ax.twiny()
    #ax_top.set_xlim(ax.get_xlim())
    #ax.set_xlabel("Forward [0->%d]" % h)
    #ax_top.set_xlabel("Backward [%d->0]" % h)
    #reverse_ticks = list(reversed(ax.get_xticklabels()))
    #ax_top.set_xticklabels(reverse_ticks)
   
    plt.title('_'.join(args.models))
    plt.savefig("plots/gradnorms_%s_areas.pdf" % '_'.join(args.models), dpi=600, bbox_inches="tight")

if "SpaceTime" in args.models:
    for h in H:
        for l in loss_based_weights[h]["forward"]:
            print ("SpaceTime %d forward gradnorm average: %.5f" % (h, l))
        for l in loss_based_weights[h]["backward"]:
            print ("SpaceTime %d backward gradnorm average: %.5f" % (h, l))
