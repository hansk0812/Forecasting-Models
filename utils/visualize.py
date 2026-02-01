from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from collections import OrderedDict

import matplotlib.colors as mcolors
from matplotlib.transforms import Bbox

import argparse

import os
import glob
import re

import numpy as np

import random

import warnings
warnings.filterwarnings("ignore")

from scipy.interpolate import make_interp_spline

import traceback

# https://www.github.com/hanskrupakar/COCO-Style-Dataset-Generator-GUI 
# (Trapezoid formula based shoelace (Gauss') formula)
def find_poly_area(coords, h):
    # find the area under the curve using its curve polygon 
    # against the ideal case of gradient norm distributions
    # across the horizon timesteps from 0 to h
    # the area is negative if the curve is below the H=0 -→ H=H line as convention
    
    index = np.where(coords[:,0] == h)[0][0]
    
    poly = coords[:index+1,:]
    
    poly = np.concatenate((poly, np.array([[
                index,
                (((coords[-1][1]-coords[0][1]))/float(coords[-1][0]-coords[0][0]))*h + float(coords[0][1])
                ]])), axis=0)
    
    line_x = np.array(list(range(index+1)))
    line_y = ((coords[-1][1]-coords[0][1])/float(coords[-1][0]-coords[0][0]))*line_x + coords[0][1]
    
    intersection_pts = np.argwhere(np.diff(np.sign(poly[:-1,1]-line_y))).flatten().tolist()

    # handle lines along lines of proportionality
    if len(intersection_pts) > 1 and h!=1:
        prev_idx = 0
        remove = []
        for idx in range(1, len(intersection_pts)):
            if intersection_pts[prev_idx] == intersection_pts[idx]-1:
                remove.append(prev_idx)
            prev_idx = idx
        
        for r in reversed(remove):
            del intersection_pts[r]

    if len(intersection_pts) > 0:
        if intersection_pts[-1] == index - 1:
            intersection_pts[-1] = index
        if len(intersection_pts) == 1 and intersection_pts[0] != 0:
            intersection_pts = [0] + intersection_pts
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
#    for idx, p in enumerate(polys):
#        if idx == 0 or len(signs) == 0:
#            if line_y[p.shape[0]//2] != p[p.shape[0]//2,1]:
#                signs.append(line_y[p.shape[0]//2] < p[p.shape[0]//2,1])
#        else:
#            signs.append(not signs[-1])
    for idx, p in enumerate(polys):
        signs.append(line_y[int(p[0][0]) + p.shape[0]//2] < p[p.shape[0]//2,1])

    return_area = 0
    for p, sign in zip(polys, signs):
        
        # coords: np.array([[x_i,y_i],...])
        x, y = p[:,0], p[:,1]
        return_area += (2*sign-1)*(0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))))/2 #shoelace algorithm
    
    return return_area, [p[-1] for p in polys]

def plot_HAM(values, model, colour_idx, plot_colors_per_model, loss_based_weights=None, h=None, cutoff_type=None):
    p, = plt.plot(np.arange(0, len(values)), values, label=model, 
                    color=plot_colors_per_model[colour_idx], linewidth=1) #0.5) 
    plt.plot([0, len(values)-1], [values[0], values[-1]], color=plot_colors_per_model[colour_idx], linestyle='--', linewidth=0.7, alpha=0.5) #0.5) 
    
    if not loss_based_weights is None:
        if cutoff_type == "forward":
            text = "H=%s: %.5f\n" % ("0  " if h < 100 else "0    ", loss_based_weights[h][cutoff_type][0])
        else:
            text =  text + "H=%d: %.5f" % (h, loss_based_weights[h][cutoff_type][0])
        
        plt.text(0.15, 0.5, text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                bbox={"facecolor": plot_colors_per_model[idx], "alpha": 0.5, "pad": 0.35, "boxstyle": "round"}, linespacing=1.5)
    
    return p

def create_rgb_colormap(color, factor=0.5, ncolors=10):
    hex_color = mcolors.to_hex(color)
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    lighter_shade = [min(r+r*factor, 255),
                     min(g+g*factor, 255),
                     min(b+b*factor, 255)]
    
    scale = np.arange(0, 1, 1./ncolors)
    
    colors = []
    for s in scale:
        color_r = min(int(r + (lighter_shade[0] - r)*s), 255)
        color_g = min(int(g + (lighter_shade[1] - g)*s), 255)
        color_b = min(int(b + (lighter_shade[2] - b)*s), 255)
        
        color_hex = mcolors.to_hex((color_r/255., color_g/255., color_b/255.))
        colors.append(color_hex)

    return list(reversed(colors))

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", help="[gradnorms, autocorr=horizon/nlags, batchsizes=(timestep1, timestep2,...)]", default="autocorr=1")
    ap.add_argument("folder", help="logs/gradnorms or logs/autocorr folder", default=None)
    ap.add_argument("--models", nargs='+', help="Models to include in visualization. Ignore for all available models.", default=None)
    ap.add_argument("--start_color_idx", default=[0], nargs='+', help="Color idx for single color per model", type=int)
    ap.add_argument("--name", help="Filename when specific layers are used in gradnorms", default=None)
    ap.add_argument("--interpolated", help="Flag to interpolate smaller gradnorms to largest gradnorm (1 per model)", action="store_true")
    args = ap.parse_args()

    assert any([x in args.mode for x in ["autocorr", "gradnorms", "batchsizes"]])
    assert "gradnorms" in args.mode or "autocorr" in args.mode
    
    assert ("gradnorms=(" in args.mode and not args.name is None) or \
           ("gradnorms" in args.mode and not '(' in args.mode) or \
           not "gradnorms" in args.mode

    assert "gradnorms" in args.mode and \
                ((any(["=(" in m for m in args.models]) and "=(" in args.mode) or \
                ("=(" in args.mode and not any(["=(" in m for m in args.models])) or \
                not '=' in args.mode) or \
            not "gradnorms" in args.mode
    
    assert (any(["=(" in m for m in args.models]) and args.name[0]=='(' and args.name[-1]==')') or \
            all([not '=' in m for m in args.models])

    assert ("gradnorms" in args.mode and args.interpolated) or not args.interpolated

    if args.interpolated:
        if not args.name is None:
            if args.name[0] == '(':
                mnames = [x.strip() for x in args.name[1:-1].split(',')]
            else:
                mnames = [args.name]
        else:
            mnames = args.models
        
        args.interpolated = {n: 0 for n in mnames}
    
    if not args.name is None and args.name[0] == '(':
        args.name = [x.strip() for x in args.name[1:-1].split(',')]

    if "=(" in args.mode:
        models_variable_name = {}
        if any(['=' in m for m in args.models]):
            for idx, m in enumerate(args.models):
                if '=' in m:
                    param_start_end = [int(x.strip()) for x in m.split('=')[1][1:-1].split(',')]
                    assert len(param_start_end) == 2

                    param_names = [x.strip() for x in args.mode.split('=')[1][1:-1].split(',')]
                    model_name = m.split('=')[0]
                    assert all([x<len(param_names) for x in param_start_end]) and param_start_end[0] <= param_start_end[1]
                    
                    if not model_name in models_variable_name:
                        models_variable_name[model_name] = {"model_idx": idx, "values": [param_start_end]}
                    else:
                        models_variable_name[model_name]["values"].append(param_start_end)
                else:
                    if not m in models_variable_name:
                        models_variable_name[m] = {"model_idx": idx, "values": [m]}
                    else:
                        models_variable_name[m]["values"].append(m)
        else:
            models_variable_name = None
    else:
        models_variable_name = None

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
    #"""
    import matplotlib as mpl
    cmap = mpl.colormaps["BuPu"] #["OrRd"]
    # Take colors at regular intervals spanning the colormap.
    colors = cmap(np.linspace(0.3, 1, 8)) #4))   
    plot_colors_per_model = np.array(colors)[args.start_color_idx]
    #"""
    # Divergent color scheme per model
    #c1 = create_rgb_colormap("#8fae22", factor=0.9, ncolors=2)
    #c2 = create_rgb_colormap("#fdb147", factor=0.9, ncolors=2)
    #plot_colors_per_model = np.concatenate((c1, c2))[args.start_color_idx]

    """
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
    """

    types = ["forward", "backward"] if "gradnorms" in args.mode else [str(h) for h in H]
    loss_based_weights = {h: {c: [] for c in types} for h in H}

    # midpoints
    midpts, midpts_plot = {}, {}
    areas = []
    for h_idx, h in enumerate(H):
        
        if "batchsizes" in args.mode:
            
            timesteps = [int(x.strip()) for x in args.mode.split('=')[-1][1:-1].split(',')]
            timesteps_h = [x for x in timesteps if x <= h]
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            
            if not all([
                os.path.exists(os.path.join(args.folder, "%s_%d_%d_batchsizes.txt" % (model, timestep, h))) \
                        for model in args.models for timestep in timesteps]):
                continue
            
            plots = []
            for idx, model in enumerate(args.models):
                model_plots = []
                for jdx, timestep in enumerate(timesteps_h):
                    if jdx == 0:
                        x1, y1, z1 = None, None, None
                    else:
                        x1, y1, z1 = xs, ys, [timesteps_h[jdx-1]]
                    
                    with open(os.path.join(args.folder, "%s_%d_%d_batchsizes.txt" % (model, timestep, h)), 'r') as f:
                        xy = [x.strip() for x in f.readlines() if x != ""]
                        
                        xs = [int(x.split(':')[0]) for x in xy]
                        
                        ys_all = [x.split(": ")[-1] for x in xy]
                        ys_all_keys, ys_all_vals = [x.split('=')[0] for x in ys_all], [float(x.split('=')[1]) for x in ys_all]
                        ys = np.array(ys_all_vals).mean()
                        #ys = [float(x.split(": ")[-1]) for x in xy]
                        
                        timestep_factor = timestep / timesteps_h[-1]
                        
                        max_y = max(ys)
                        ys = [y/max(ys) for x, y in zip(xs, ys)]
                        #ys = [(y*x**1.1)/max(ys) for x, y in zip(xs, ys)]
                        #ys = [(y*x**1.5*timestep_factor**0.1 * np.cos(timestep/y))/max(ys) for x, y in zip(xs, ys)]
                        
                      
                        if jdx > 0:
                            ax.fill_between(x1, y1, z1, xs, ys, [timestep], alpha=0.8, edgecolor=plot_colors_per_model[idx],
                                    facecolors=create_rgb_colormap(plot_colors_per_model[idx], 0.4, len(xs)-1))
                        
                        model_label = model
                        #model_label = "%s (Scale = %.5f)" % (model, max(ys))
                        p, = ax.plot(xs, ys, zs=timestep, label=model_label, color=plot_colors_per_model[idx])

                        bbox_props = dict(boxstyle="round,pad=0.1", fc="cyan", ec="b", lw=1)
                        ax.text(xs[-1]+200, min(ys) - min(0.05, min(ys))*min(ys), timestep+10, "S=%.3f" % max_y, ha='center', va='baseline', 
                                    fontsize=4, bbox=bbox_props, in_layout=True) #-min(ys)*min(ys)
 
                        model_plots.append(p)
                    plots.append(tuple(model_plots))

            #ax.set_xlim(0, 1)
            #ax.set_ylim(0, 1)
            #ax.set_zlim(0, 1)
            
            ax.view_init(elev=20., azim=45, roll=0)

            plt.legend(plots, args.models, loc=(0.4, 0.25), handler_map={tuple: HandlerTuple(ndivide=None)})
            
            plt.xlabel("Batch Size", labelpad=-3)
            plt.ylabel("Interpolated Gradnorm Avg", labelpad=-5)
            ax.set_zlabel("Timestep", in_layout=True, verticalalignment="baseline", labelpad=0.5)
            plt.title("Gradnorm averages by batch sizes per model", y=0.87, x=0.45)
            
            plt.tick_params(axis='y', which="both", pad=-2)
            plt.tick_params(axis='x', which="both", pad=-2)

            bbox = fig.get_tightbbox()
            bbox_pts = bbox.get_points()
            bbox_pts[0][0] -= bbox_pts[0][0]*0.11
            bbox_pts[1][0] -= bbox_pts[1][0]*0.017
            bbox_pts[0][1] += bbox_pts[0][1]*0.1
            bbox_pts[1][1] -= bbox_pts[1][1]*0.105
            bbox = Bbox(bbox_pts)
            
            plt.savefig("plots_/gradnorms_%s_%d_batchsizes.pdf" % ('_'.join(args.models), h), dpi=300, bbox_inches=bbox)
            
            plt.clf()

            if timesteps[-1] == h:
                exit()

            continue

        fig, ax = plt.subplots()
    
        if "autocorr" in args.mode:
            autocorrs_gt = []
        else:
            poly_areas = OrderedDict({k: {} for k in types})
            poly_areas_pts = OrderedDict({k: {} for k in types})

        plot_diffs = OrderedDict()
        point_plts = []
        for cutoff_type in types:
            
            fnames = sorted(glob.glob(os.path.join(args.folder, "*_%d_%s_%s.txt" % (
                h, cutoff_type, args.mode.split('=')[0]))))

            for idx in range(len(fnames)-1,-1,-1):
                if not args.models is None and not any([fnames[idx].split('/')[-1].split('_')[0] == m.split('=')[0] for m in args.models]):
                    del fnames[idx]
            
            if len(fnames) == 0:
                print ("H=%d files not found!" % h if "gradnorms" in args.mode else int(h/int(cutoff_type))); continue #exit()
            
            fnames = sorted(fnames, key=lambda x: x.split('/')[-1].split(".txt")[0])
            
            # Eliminate duplicates for code to handle it!
            fnames = list(dict.fromkeys(fnames))
            
            # Optionally add table to plot
            table = None
            for idx, fname in enumerate(fnames):
                model = fname.split('/')[-1].split('_')[0]
                
                if not model in plot_diffs and cutoff_type == "forward":
                    if any([model in x for x in args.models]):
                        indices = [idx for idx, m in enumerate(args.models) if model in m]
                        for index in indices:
                            if not '=' in args.models[index]:
                                plot_diffs[model] = {}
                            else:
                                plot_diffs[args.name[index]] = {}
                
                if "gradnorms" in args.mode:
                    
                    """ # Optionally add table to plot
                    if table is None:
                        from matplotlib.font_manager import FontProperties
                        table = plt.table(cellText=[
                                    #["Pyraformer", "0.44", "0.49"], 
                                    #["EncReversedMask", "0.37", "0.44"], 
                                    #["ReversedMask", "0.42", "0.48"], 
                                    #["WithoutMask", "0.46", "0.50"]
                                    #["FEDformerCE22", "0.20", "0.33"],
                                    #["FEDformerCE10", "", ""],
                                    
                                    #["20", "0.224", "0.341"],
                                    #["30", "0.227", "0.345"],
                                    #["40", "0.229", "0.347"],
                                    #["50", "0.233", "0.350"]
                                    
                                    #["190", "0.49", "0.53"],
                                    #["40", "0.44", "0.49"]
                                    
                                    #["EncReversedMask", "0.37", "0.44"], 
                                    #["Pyraformer", "0.44", "0.49"], 
                                    
                                    ["Without Roll", "40", "0.218", "0.331"],
                                    ["Autoformer", "190", "0.227", "0.340"]
                                    ],
                                    colLabels=["Model", "Batch", "MSE", "MAE"], #['Epochs', 'MSE', 'MAE'],
                                    loc='upper center',
                                    cellLoc='center',
                                    bbox=[0.3, 0.84, 0.5, 0.15])
                                    #bbox=[0.3, 0.84, 0.45, 0.15])
                        table.auto_set_font_size(False)
                        table.set_fontsize(10)
                        table.set_alpha(0.6)
                        for (row, col), cell in table.get_celld().items():
                            if (row == 0):
                                cell.set_text_props(fontproperties=FontProperties(weight='bold'))
                            else:
                                cell.set_text_props(color=plot_colors_per_model[row-1])
                            if col > 0:
                                cell.set(width=0.1)
                            else:
                                cell.set(width=0.16)
                    """

                    if not model in plots:
                        for mdl in [m for m in args.models if model in m]:
                            if not '=' in mdl:
                                plots[model] = []
                                midpts[model] = []

                    with open(fname, 'r') as f:
                        values, min_idx = [], 1e7
                        for jdx, line in enumerate(f.readlines()):
                            if model.lower() == "spacetime" and "Gra " in line:
                                pt = float(line.split(": ")[-1])
                                loss_based_weights[h][cutoff_type].append(pt)

                            if not "Grad " in line:
                                continue
                            
                            gradnorm_str = line.split(": ")[-1]
                            if not '=' in gradnorm_str:
                                values.append(float(gradnorm_str))
                            else:
                                if jdx < min_idx:
                                    min_idx = jdx
                                    keys = [x.split('=')[0] for x in gradnorm_str.split(' ')]
                                values.append([float(x.split('=')[1].strip()) for x in gradnorm_str.split(' ')])
                    
                    # values: (h,l) horizon x num_layers
                    # ASSUMPTION: Interpolation independently is the same as interpolation of the mean
                    # ALSO TRANSPOSES VALUES
                    if len(values) != h + 1:
                        if isinstance(values[0], float):
                            x_range = np.arange(0, h+1, np.round(h, -2)/(len(values)-2)).tolist() + [h]
                            # unpredictable cubic interpolation when sequence is decreasing; reverse
                            if cutoff_type != "forward":
                                values = list(reversed(values))
                            values_spline = make_interp_spline(x_range, values)
                            X = np.linspace(0, h, h+1)
                            values = values_spline(X)
                            if cutoff_type != "forward":
                                values = values[::-1]
                        else:
                            values = np.array(values)

                            # DECISION: Interpolating one feature at a time gives very different plots (#TODO: Future Work)
                            if not '=' in args.mode:
                                values = values.mean(axis=1)[:,np.newaxis]
                            else:
                                values = values

                            values_interpolated = []
                            for jdx in range(len(values[0])):
                                # unpredictable cubic interpolation when sequence is decreasing; reverse
                                if cutoff_type != "forward":
                                    values_s = values[:,jdx][::-1]
                                else:
                                    values_s = values[:,jdx]
                                x_range = np.arange(0, h+1, np.round(h, -2)/(len(values_s)-2)).tolist() + [h]
                                values_spline = make_interp_spline(x_range, values_s)
                                X = np.linspace(0, h, h+1)
                                if cutoff_type != "forward":
                                    values_sy = values_spline(X)[::-1]
                                else:
                                    values_sy = values_spline(X)
                                values_interpolated.append(values_sy)
                            values = values_interpolated
                    else:
                        values = np.array(values).T
                    
                    if '=' in args.mode:
                        if not models_variable_name is None and \
                            model in models_variable_name and \
                            isinstance(models_variable_name[model]["values"], list):
                            all_values = np.array(values).mean(axis=0)
                        else:
                            all_values = None
                        
                        # Sort and remove values from file's layer keys to gradnorm's layer keys
                        if '(' in args.mode:
                            values_new = []
                            keys_diff_models = [x.strip() for x in args.mode.split('=')[-1][1:-1].split(',')]
                            for key in keys_diff_models:
                                key_idx = keys.index(key)
                                values_new.append(values[key_idx])
                            values = np.array(values_new)
                        else:
                            key = args.mode.split('=')[-1]
                            key_idx = keys.index(key)
                            values = values[key_idx]
                    
                    if not isinstance(values[0], float):
                        if not models_variable_name is None and isinstance(models_variable_name[model]["values"], list):
                            legend_list = {}
                            
                            if model in models_variable_name[model]["values"]:
                                if cutoff_type == "forward":
                                    print (model, "max value:", max(all_values), "interpolation factor:", 1./max(all_values))
                                
                                if isinstance(args.interpolated, dict):
                                    if args.interpolated[model] == 0:
                                        args.interpolated[model] = 1./max(all_values)
                                        model_legend = model + " (%.4f)" % max(all_values)
                                    else:
                                        fc = 1./args.interpolated[model]
                                        model_legend = model + " (%.4f)" % fc
                                    all_values *= args.interpolated[model] #/= max(all_values)
                                else:
                                    model_legend = model

                                p = plot_HAM(all_values, model_legend, idx, plot_colors_per_model)
                                legend_list[model_legend] = [p]
                            
                                model_index = models_variable_name[model]["values"].index(model)
                            else:
                                model_index = 1e7
                            
                            # values: (l,h) --> (m,h) num layers in file x horizon --> num models in file x horizon
                            values_many = []
                            for jdx, start_end in enumerate(models_variable_name[model]["values"]):
                                if jdx == model_index:
                                    values_many.append(all_values)
                                else:
                                    start, end = start_end
                                    values_many.append(np.array(values[start:(end+1)]).mean(axis=0))
                            values = np.array(values_many)
                            
                            for jdx, value in enumerate(values):

                                if jdx == model_index:
                                    continue
                                
                                if cutoff_type == "forward":
                                    print (args.name[models_variable_name[model]["model_idx"]+jdx], "max value:", max(value), "interpolation factor:", 1./max(value))
                                
                                if isinstance(args.interpolated, dict):
                                    if args.interpolated[args.name[models_variable_name[model]["model_idx"]+jdx]] == 0:
                                        args.interpolated[args.name[models_variable_name[model]["model_idx"]+jdx]] = 1./max(value)
                                        model_legend = args.name[models_variable_name[model]["model_idx"]+jdx] + " (%.4f)" % max(value)
                                    else:
                                        fc = 1./args.interpolated[args.name[models_variable_name[model]["model_idx"]+jdx]]
                                        model_legend = args.name[models_variable_name[model]["model_idx"]+jdx] + " (%.4f)" % fc
                                    value *= args.interpolated[args.name[models_variable_name[model]["model_idx"]+jdx]] #/= max(value)
                                else:
                                    model_legend = args.name[models_variable_name[model]["model_idx"]+jdx]

                                p = plot_HAM(value, model_legend, idx, plot_colors_per_model)
                                if model_legend in legend_list:
                                    legend_list[model_legend].append(p)
                                else:
                                    legend_list[model_legend] = [p]
                        else:
                            
                            values = np.array(values).mean(axis=0)
                            if cutoff_type == "forward":
                                print (model, "max value:", max(values), "interpolation factor:", 1./max(values))
                            
                            if isinstance(args.interpolated, dict):
                                if args.interpolated[model] == 0:
                                    args.interpolated[model] = 1./max(values)
                                    model_legend = model + " (%.4f)" % max(values)
                                else:
                                    fc = 1./args.interpolated[model] 
                                    model_legend = model + " (%.4f)" % fc
                                values *= args.interpolated[model]
                            else:
                                model_legend = model

                            if model != "SpaceTime":
                                p = plot_HAM(values, model_legend, idx, plot_colors_per_model)
                            else:
                                p = plot_HAM(values, model_legend, idx, plot_colors_per_model, loss_based_weights, h, cutoff_type)

                    else:
                            if cutoff_type == "forward":
                                print (model, "max value:", max(values), "interpolation factor:", 1./max(values))
                            if isinstance(args.interpolated, dict):
                                if args.interpolated[model] == 0:
                                    args.interpolated[model] = 1./max(values)
                                    model_legend = model + " (%.4f)" % max(values)
                                else:
                                    fc = 1./args.interpolated[model] 
                                    model_legend = model + " (%.4f)" % fc
                                values *= args.interpolated[model]
                            else:
                                model_legend = model

                            #values = np.array(values).mean(axis=0)
                            if model != "SpaceTime":
                                p = plot_HAM(values, model_legend, idx, plot_colors_per_model)
                            else:
                                p = plot_HAM(values, model_legend, idx, plot_colors_per_model, loss_based_weights, h, cutoff_type)
                    
                    if len(values.shape) > 1:
                        model_names_multiple = args.models[models_variable_name[model]["model_idx"]: models_variable_name[model]["model_idx"]+len(values)]
                        model_names_multiple = [x if not '=' in x else args.name[models_variable_name[model]["model_idx"]+jdx] for jdx, x in enumerate(model_names_multiple)]
                        for n, v in zip(model_names_multiple, values):
                            if n in plot_diffs:
                                plot_diffs[n][cutoff_type] = v
                            else:
                                plot_diffs[n] = {cutoff_type: v}
                    else:
                        plot_diffs[model][cutoff_type] = values

                    #if model.lower() == "spacetime":
                    #    plt.plot(0, first_pt, marker='o', color=plot_colors_per_model[idx])
                    
                    # calculate area
                    if len(values.shape) > 1:
                        for n, v in zip(model_names_multiple, values):
                            poly_areas[cutoff_type][n] = []
                            pts = np.stack((np.arange(0, len(v)), v)).transpose()
                            for h_ in range(1, h+1):
                                #plt.plot([0, len(values)-1], [values[0], values[-1]], color=plot_colors[idx], linestyle='--')
                                #plt.plot(list(range(h_)), values[:h_], color=plot_colors[idx], linestyle='--')
                                area, intersect_pts = find_poly_area(pts, h_)
                                poly_areas[cutoff_type][n].append(area)
                                
                                if len(intersect_pts) > 2:
                                    poly_areas_pts[cutoff_type][n] = intersect_pts[1:-1]
                                else:
                                    poly_areas_pts[cutoff_type][n] = []
                    
                    else:
                        poly_areas[cutoff_type][model] = []
                        pts = np.stack((np.arange(0, len(values)), values)).transpose()
                        for h_ in range(1, h+1):
                            #plt.plot([0, len(values)-1], [values[0], values[-1]], color=plot_colors[idx], linestyle='--')
                            #plt.plot(list(range(h_)), values[:h_], color=plot_colors[idx], linestyle='--')
                            area, intersect_pts = find_poly_area(pts, h_)
                            poly_areas[cutoff_type][model].append(area)
                            
                            if len(intersect_pts) > 2:
                                poly_areas_pts[cutoff_type][model] = intersect_pts[1:-1]
                            else:
                                poly_areas_pts[cutoff_type][model] = []
                    
                    if len(values.shape) > 1:
                        for n, v in zip(model_names_multiple, values):
                            if not n in midpts:
                                midpts[n] = []
                            if len(midpts[n]) == 0:
                                midpts[n].append(np.array(v))
                            else:
                                midpts[n] = np.argwhere(np.diff(np.sign(np.array(v)-midpts[n][0])))
                                
                        plots.update(legend_list)
                    else:
                        if len(midpts[model]) == 0:
                            midpts[model].append(np.array(values))
                        else:
                            midpts[model] = np.argwhere(np.diff(np.sign(np.array(values)-midpts[model][0])))
                        
                        if '(' in model_legend and model_legend.split(" (")[0] in plots:
                            plots[model_legend] = plots.pop(model_legend.split(" (")[0])
                        plots[model_legend].append(p)
                
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
                plt.legend(prop={"size": 10}, loc="best")
                plt.xlabel("H=%d lags" % h, fontsize=10)
                plt.ylabel("Autocorrelation", fontsize=10)
                plt.title("Self attention models' ACF averages over the test set", fontsize=10)
                
                if any(['=' in m for m in args.models]):
                    models_name = '_'.join(args.name)
                else:
                    models_name = '_'.join(sorted(args.models))
                plt.savefig("plots_/autocorrs_%d_%s.pdf" % (h, "all_models" if args.models is None else models_name), dpi=300, bbox_inches="tight")
                #plt.show()
                plt.clf()

            if "gradnorms" in args.mode:
                plt.legend([tuple(plots[model]) for model in plots], list(plots.keys()),
                        handler_map={tuple: HandlerTuple(ndivide=None)}, prop={"size": 10}, loc="best") #6})#, loc="top")
        
        if "gradnorms" in args.mode:
            ax_top = ax.twiny()
            ax_top.set_xticks([])
            ax_top.set_xlabel("Anti-Causal Sub-Series: x→%d" % h, fontsize=10)

#            ax_top.set_xlim(ax.get_xlim())
#            ax.set_xlabel("Forward [0→%d]" % h)
#            ax_top.set_xlabel("Anti-Causal Sub-Series: x → x_reverse)")
#            reverse_ticks = list(reversed(ax.get_xticklabels()))
#            extent = int(reverse_ticks[0]._x - reverse_ticks[1]._x)
##            
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
#                text_obj._text = str(h) #str(int(reverse_ticks[start_idx + idx]._x)-extent)
#                text_obj._x = h #int(text_obj._text)
#            ax_top.set_xticklabels(reverse_ticks)
            
            #ax.set_ylim(top=0.009)

            ax.set_xlabel("Causal Sub-Series: 0→x", fontsize=10)
            
            if not args.name is None:
                min_str_length = min([len(x) for x in args.name])
                for index in range(min_str_length):
                    if any([args.name[jdx-1][index] != args.name[jdx][index] for jdx in range(1, len(args.name))]):
                        break
                common_name_str = args.name[0][:index]
            
            ylabel = "Gradient Norm Average" if args.name is None else "%s Gradient Norm Average" % common_name_str
            if isinstance(args.interpolated, dict):
                ylabel = "Interpolated " + ylabel
            ax.set_ylabel(ylabel, fontsize=10)

            if '=' in args.mode:
                if '(' in args.mode:
                    gradnorms_str = "gradnorms_" + common_name_str
                    #'_'.join([x.strip() for x in args.mode.split('=')[1:-1].split(',')])
                else:
                    gradnorms_str = "gradnorms_" + args.mode.split('=')[-1].replace('.', '-')
            else:
                gradnorms_str = "gradnorms"

            if any(['=' in m for m in args.models]):
                models_name = '_'.join(args.name)
            else:
                models_name = '_'.join(sorted(args.models))
            print ("saving to ", "plots_/%s_%d_%s.pdf" % (gradnorms_str, h, "all_models" if args.models is None else models_name))
            plt.savefig("plots_/%s_%d_%s.pdf" % (gradnorms_str, h, "all_models" if args.models is None else models_name), dpi=300, bbox_inches="tight")
            
            #plt.show(); exit()
            
            plt.clf()
            min_y, midpts_diff = np.inf, {}

            #plt.plot([0, h], [0, 0], color='black')
            plt.plot(list(range(h+1)), np.arange(-1, 1, 2./(h+1)), linestyle="--", color='black', label="Uniform Differences Line")
            
            for idx, model_name in enumerate(sorted(plot_diffs.keys())):
                diff = np.array(plot_diffs[model_name]["forward"]) - np.array(plot_diffs[model_name]["backward"])
                diff /= diff.max()
                
                plt.plot(list(range(len(diff))), diff, label=model_name, color=plot_colors_per_model[idx])

                min_y = np.min(np.array([min_y, diff.min()]))
                midpts_diff[model_name] = np.abs(diff).argmin()
            
            for idx, model_name in enumerate(sorted(midpts_diff.keys())):
                plt.plot(midpts_diff[model_name], min_y, marker='o', markersize=3, 
                            color=plot_colors_per_model[idx])
                
                # Lines of Proportionality dependent on equivariant point in Difference plots
                plt.plot([midpts_diff[model_name], h], [0, 1], linestyle="--", color=plot_colors_per_model[idx], alpha=0.35)
                plt.plot([0, midpts_diff[model_name]], [-1, 0], linestyle="--", color=plot_colors_per_model[idx], alpha=0.35)
            
            plt.legend(prop={"size": 10}, loc="best")
            #plt.title("Difference between forward and backward mode gradient norm averages")
            plt.xlabel("Timestep h for subseries", fontsize=10)
            ylabel = "Difference at h [g(0→h) - g(h→%d)]" % h if args.name is None else "%s Difference at h [g(0→h) - g(h→%d)]" % (common_name_str, h)
            
            if isinstance(args.interpolated, dict):
                ylabel = "Interpolated " + ylabel
            plt.ylabel(ylabel, fontsize=10)
            plt.title("Differences plots over %d models" % len(args.models), fontsize=10)
            
            if '=' in args.mode:
                if '(' in args.mode:
                    gradnorms_str = "gradnorms_" + common_name_str 
                    #'_'.join([x.strip() for x in args.mode.split('=')[1:-1].split(',')])
                else:
                    gradnorms_str = "gradnorms_" + args.mode.split('=')[-1].replace('.', '-')
            else:
                gradnorms_str = "gradnorms"
            
            if any(['=' in m for m in args.models]):
                models_name = '_'.join(args.name)
            else:
                models_name = '_'.join(sorted(args.models))
            plt.savefig("plots_/%s_%d_%s_diffs.pdf" % (gradnorms_str, h, "all_models" if args.models is None else models_name), dpi=300, bbox_inches="tight")
            plt.clf()
            
            """
            # Heatmaps slightly harder to interpret between models
            for idx, model_name in enumerate(sorted(plot_diffs.keys())):
                heatmap = np.zeros((h//2,h+1))
                for jdx in range(h//2):
                    diff = np.array(plot_diffs[model_name]["forward"])[:h-idx+1] - np.array(plot_diffs[model_name]["backward"])[idx:]
                    heatmap[jdx][idx:] = diff
                plt.imshow(heatmap, cmap='hot', interpolation='nearest')
                plt.title("%s Heatmap of Differences over the first h→h//2 values" % model_name)
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
            
            area_plts, legend_labels = [], []
            for idx, model in enumerate(poly_areas[cutoff_type].keys()):
                plt.plot([0, len(poly_areas[cutoff_type][model])], [0, 0], color="black")
                
                #max_area = max([max(poly_areas[c][model]) for c in types])
                area_plt_pair = []
                for cutoff_type in types:
                    p, = plt.plot(np.arange(0, len(poly_areas[cutoff_type][model])),
                               #poly_areas[cutoff_type][model] / max_area, label=model + "[%s→%s]" % (
                               poly_areas[cutoff_type][model], label=model + "[%s→%s]" % (
                                    "0" if cutoff_type=="forward" else "x",
                                    "x" if cutoff_type=="forward" else str(len(poly_areas[cutoff_type][model]))), 
                                color=plot_colors_per_model[idx],
                                linestyle="dashed" if cutoff_type==types[1] else "dotted")
                    area_plt_pair.append(p)
                legend_labels.append(model)

                area_plts.append(area_plt_pair)
                    
                try:
                    if len(poly_areas_pts[cutoff_type][model]) > 0:
                        for pt in poly_areas_pts[cutoff_type][model]:
                            p, = plt.plot(pt[0], poly_areas[cutoff_type][model][int(pt[0])], 'o', markersize=3, 
                                            color=plot_colors_per_model[idx], label="Intersection with Line of Proportionality")
                            point_plts.append(p)
                except Exception:
                    traceback.print_exc()
                    pass
                
            if len(area_plts) > 0:
                markers = [tuple(m) for m in zip(*area_plts)]
                min_max_idxs = [np.argmin(args.start_color_idx), np.argmax(args.start_color_idx)]
                markers = [(m[min_max_idxs[0]],m[min_max_idxs[1]]) for m in markers]
                if len(point_plts) > 0:
                    markers.append(tuple(point_plts))
                legend_line_type = plt.legend(markers, ["Causal [0→x]", "Anti-Causal [x→H]"] if len(point_plts)==0 \
                                                    else ["Causal [0→x]", "Anti-Causal [x→H]", "Intersection with Line of Proportionality"],
                                                prop={"size": 10}, loc=0,
                                                handler_map={tuple: HandlerTuple(ndivide=None)})
                legend_models = plt.legend([l[1] for l in area_plts], legend_labels, prop={"size": 10}, loc=6)
                ax.add_artist(legend_line_type)
                ax.add_artist(legend_models)
            
            #ax_top = ax.twiny()
            #ax_top.set_xlim(ax.get_xlim())
            #ax.set_xlabel("Forward [0→%d] (dotted line)" % h)
            #ax_top.set_xlabel("Backward [%d→0] (dashed line)" % h)
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
            
            #plt.ylim(top=0.75)
            
            plt.xlabel("Timesteps", fontsize=10)
            ylabel = "Signed Area w.r.t line of proportionality" if args.name is None else \
                     "%s Signed Area w.r.t line of proportionality" % common_name_str
            
            if isinstance(args.interpolated, dict):
                ylabel = "Interpolated " + ylabel
            plt.ylabel(ylabel, fontsize=10)
            plt.title("Dotted Causal mode areas and Dashed Anti-Causal mode areas", fontsize=10)
            
            if '=' in args.mode:
                if '(' in args.mode:
                    gradnorms_str = "gradnorms_" + common_name_str 
                    #'_'.join([x.strip() for x in args.mode.split('=')[1:-1].split(',')])
                else:
                    gradnorms_str = "gradnorms_" + args.mode.split('=')[-1].replace('.', '-')
            else:
                gradnorms_str = "gradnorms"

            if any(['=' in m for m in args.models]):
                models_name = '_'.join(args.name)
            else:
                models_name = '_'.join(sorted(args.models))
            plt.savefig("plots_/%s_%d_%s_areas.pdf" % (gradnorms_str, h, "all_models" if args.models is None else models_name), dpi=300, bbox_inches="tight")
            #plt.show()
            plt.clf()

            areas.append(poly_areas)
        #areas.append(poly_areas)

    if not "gradnorms" in args.mode:
        exit()
    
    plt.clf()

    midpts_flag = True
    label_plots = []
    for idx, k in enumerate(midpts_plot):
        if len(midpts_plot[k]) != 4:
            midpts_flag = False
            break
        p, = plt.plot(H, midpts_plot[k], label=k, marker='o', color=plot_colors_per_model[idx], linewidth=1) #0.5) 
        label_plots.append(p)
    
    if midpts_flag:
 
        label_plots = []
        for idx, k in enumerate(midpts_plot):
            p, = plt.plot(H, midpts_plot[k], label=k, marker='o', color=plot_colors_per_model[idx], linewidth=1) #0.5) 
            label_plots.append(p)

        for h in H:
            p, = plt.plot([3*h/4, 5*h/4], [h/2, h/2], linestyle="--", color='black', label="H/2 Line")
            label_plots.append(p)
    
        label_keys = list(midpts_plot.keys())
        label_keys.append("H/2 Line")
        plt.legend(label_plots, label_keys, handler_map={tuple: HandlerTuple(ndivide=None)}, prop={"size": 10}, loc="best")
        plt.title("Gradient equivariance points over %d models" % len(midpts_plot))
        
        plt.xticks(H, ["H=%d" % H[idx] for idx in range(len(H))])
        
        plt.xlabel("Model Horizon Sizes", fontsize=10)
        plt.ylabel("Timesteps in the Horizon", fontsize=10)
        
        if '=' in args.mode:
            if '(' in args.mode:
                gradnorms_str = "gradnorms_" + common_name_str 
                #'_'.join([x.strip() for x in args.mode.split('=')[1:-1].split(',')])
            else:
                gradnorms_str = "gradnorms_" + args.mode.split('=')[-1].replace('.', '-')
        else:
            gradnorms_str = "gradnorms"

        if any(['=' in m for m in args.models]):
            models_name = '_'.join(args.name)
        else:
            models_name = '_'.join(sorted(args.models))

        if args.interpolated is None:
            plt.savefig("plots_/%s_%d_%s_midpts.pdf" % (gradnorms_str, h, "all_models" if args.models is None else models_name), dpi=300, bbox_inches="tight")
        #plt.show()

    # Plot interpolated area plot for H=720 model only!
    if not areas[0]["forward"] and not areas[1]["forward"] and not areas[2]["forward"]:
    
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
            
            plots_legend = [[] for _ in range(len(area_dict["forward"]))]
            for k in area_dict:
                for jdx, m in enumerate(area_dict[k]):
                    p, = plt.plot(np.arange(0, 1, 1/len(area_dict[k][m])), 
                                  area_dict[k][m]/max(maxs[m]), 
                                  label=m, 
                                  color=plot_colors_per_model[jdx],
                                  linestyle="dashed" if k=="backward" else "dotted")
                    plots_legend[jdx].append(p)

        plots_legend = [tuple(x) for x in plots_legend]
        plt.legend(plots_legend, area_dict["forward"], prop={"size": 10}, loc="best",
                   handler_map={tuple: HandlerTuple(ndivide=None)})

        #ax_top = ax.twiny()
        #ax_top.set_xlim(ax.get_xlim())
        #ax.set_xlabel("Forward [0→%d]" % h)
        #ax_top.set_xlabel("Backward [%d→0]" % h)
        #reverse_ticks = list(reversed(ax.get_xticklabels()))
        #ax_top.set_xticklabels(reverse_ticks)
       
        plt.title("Interpolated area plot over %s models" % ", ".join(args.models), fontsize=10)
        plt.xlabel("Fraction of Horizon", fontsize=10)
        plt.ylabel("Fraction of gradient norm average", fontsize=10)
        plt.savefig("plots_scores/gradnorms_%s_areas.pdf" % '_'.join(args.models), dpi=300, bbox_inches="tight")
        
        exit()

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
                l = "[ %s  →  %s ]" % (
                        "0".rjust(9) if k=="forward" else ("x*%d"%h).rjust(6 if h>100 else 7), 
                        ("%d"%h).rjust(6 if h>100 else 7) if k=="backward" else ("x*%d"%h).rjust(5 if h>100 else 6))
                
                plt.plot(np.arange(0, 1, 1/len(area_dict[k][m])), 
                         area_dict[k][m]/max(maxs[m]), 
                         label=l, 
                         color=plot_colors[idx],
                         linestyle="dashed" if k=="backward" else "dotted")
            
    plt.legend(prop={"size": 10}, loc="best")
    #ax_top = ax.twiny()
    #ax_top.set_xlim(ax.get_xlim())
    #ax.set_xlabel("Forward [0→%d]" % h)
    #ax_top.set_xlabel("Backward [%d→0]" % h)
    #reverse_ticks = list(reversed(ax.get_xticklabels()))
    #ax_top.set_xticklabels(reverse_ticks)
   
    plt.title("Interpolated area plots over %s H={96,192,336,720} models" % args.models[0], fontsize=10)
    plt.xlabel("Fraction of Horizon", fontsize=10)
    ylabel = "Fraction of gradient norm average" if args.name is None else \
             "%s Fraction of gradient norm average" % common_name_str
    plt.ylabel(ylabel, fontsize=10)
   
    if '=' in args.mode:
        if '(' in args.mode:
            gradnorms_str = "gradnorms_" + common_name_str 
            #'_'.join([x.strip() for x in args.mode.split('=')[1:-1].split(',')])
        else:
            gradnorms_str = "gradnorms_" + args.mode.split('=')[-1].replace('.', '-')
    else:
        gradnorms_str = "gradnorms"
    
    if any(['=' in m for m in args.models]):
        models_name = '_'.join(args.name)
    else:
        models_name = '_'.join(sorted(args.models))
    plt.savefig("plots_/%s_%s_areas.pdf" % (gradnorms_str, "all_models" if args.models is None else models_name), dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots()
    plt.plot([0, 1], [0, 0], color="black")

    #print (len(areas), areas[0].keys(), areas[0]["forward"]); exit()
    exit()

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
                l = "[ %s  →  %s ]" % (
                        "0".rjust(9) if k=="forward" else ("x*%d"%h).rjust(6 if h>100 else 7), 
                        ("%d"%h).rjust(6 if h>100 else 7) if k=="backward" else ("x*%d"%h).rjust(5 if h>100 else 6))
                
                plt.plot(np.arange(0, 1, 1/len(area_dict[k][m])), 
                         area_dict[k][m]/max(maxs[m]), 
                         label=l, 
                         color=plot_colors[idx],
                         linestyle="dashed" if k=="backward" else "dotted")
            
    plt.legend(prop={"size": 10}, loc="best")
    #ax_top = ax.twiny()
    #ax_top.set_xlim(ax.get_xlim())
    #ax.set_xlabel("Forward [0→%d]" % h)
    #ax_top.set_xlabel("Backward [%d→0]" % h)
    #reverse_ticks = list(reversed(ax.get_xticklabels()))
    #ax_top.set_xticklabels(reverse_ticks)
   
    plt.title("Interpolated area plots over %s H={96,192,336,720} models" % args.models[0], fontsize=10)
    plt.xlabel("Fraction of Horizon", fontsize=10)
    ylabel = "Fraction of gradient norm average" if args.name is None else \
             "%s Fraction of gradient norm average" % common_name_str
    
    if isinstance(args.interpolated, dict):
        ylabel = ylabel.replace("gradient", "interpolated gradient")
    plt.ylabel(ylabel, fontsize=10)
    if '=' in args.mode:
        if '(' in args.mode:
            gradnorms_str = "gradnorms_" + common_name_str
            #'_'.join([x.strip() for x in args.mode.split('=')[1:-1].split(',')])
        else:
            gradnorms_str = "gradnorms_" + args.mode.split('=')[-1].replace('.', '-')
    else:
        gradnorms_str = "gradnorms"
    
    if any(['=' in m for m in args.models]):
        models_name = '_'.join(args.name)
    else:
        models_name = '_'.join(sorted(args.models))
    plt.savefig("plots_/%s_%s_areas.pdf" % (gradnorms_str, "all_models" if args.models is None else models_name), dpi=300, bbox_inches="tight")

if "SpaceTime" in args.models:
    for h in H:
        for l in loss_based_weights[h]["forward"]:
            print ("SpaceTime %d forward gradnorm average: %.5f" % (h, l))
        for l in loss_based_weights[h]["backward"]:
            print ("SpaceTime %d backward gradnorm average: %.5f" % (h, l))

