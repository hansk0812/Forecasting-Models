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
        #if intersection_pts[-1] == index - 1:
        #    intersection_pts[-1] = index
        if intersection_pts == [0]:
            intersection_pts.append(index)
        else:
            intersection_pts = [intersection_pts[idx] if idx == 0 else intersection_pts[idx]+1 for idx in range(len(intersection_pts))] + [index]
    else:
        intersection_pts = [0, index]
    
    polys, poly_adds = [], []
    for idx in range(len((intersection_pts))-1):
        p = poly[intersection_pts[idx]:intersection_pts[idx+1]+1,:].tolist()
        if idx > 1:
            start_pt = poly[intersection_pts[idx-1]][0].astype(np.int32)
            p = [[line_x[start_pt], line_y[start_pt]]] + p + [[line_x[intersection_pts[idx+1]], line_y[intersection_pts[idx+1]]]]
            poly_adds.append(2)
        else:
            p = p + [[line_x[-1], line_y[-1]]]
            poly_adds.append(1)
        polys.append(np.array(p))
    
    positions = [0] + [p.shape[0]-a for (p, a) in zip(polys[:-1], poly_adds[:-1])]
    for idx in range(1, len(positions)):
        positions[idx] = positions[idx-1]+positions[idx]
    
    #print (line_y, poly)
    #if len(positions)>1:
    #    print (positions[1], polys[1].shape[0]) # 65, 2
    #print ('y index, poly index', [[positions[idx] + (-poly_adds[idx]+p.shape[0])//2, line_y, p.shape[0]//2, p] for idx, p in enumerate(polys)]) #p.shape[0]-1 maybe
    signs = [(p.shape[0]//2==0 and p[1][-1] < line_y[1]) or line_y[min(len(line_y)-1, positions[idx] + (-poly_adds[idx]+p.shape[0])//2)] < p[p.shape[0]//2,1] \
                for idx, p in enumerate(polys)]

    return_area = 0
    for p, sign in zip(polys, signs):
        #print (sign)
        #plt.plot(line_x, line_y)
        #plt.plot(p[:,0], p[:,1])
        # coords: np.array([[x_i,y_i],...])
        x, y = p[:,0], p[:,1]
        return_area += (2*sign-1)*(0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))))/2 #shoelace algorithm

    #print ()
    #plt.show()
    return return_area

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

            lines_f.append(lines_b[0])
            lines_b.append(lines_f[0])
            
            with open(f_f, 'w') as f:
                f.writelines(lines_f)
            with open(f_b, 'w') as f:
                f.writelines(lines_b)

    # midpoints
    midpts, midpts_plot = {}, {}

    for h_idx, h in enumerate(H):

        #if h < 720:
        #    continue

        types = ["forward", "backward"] if args.mode == "gradnorms" else [str(int(h/int(args.mode.split('=')[-1])))]
        
        if "autocorr" in args.mode:
            autocorrs_gt = []
        else:
            poly_areas = OrderedDict({k: {} for k in types})

        for cutoff_type in types:
            
            #if cutoff_type == "forward":
            #    continue

            print ()

            fnames = sorted(glob.glob(os.path.join(args.folder, "*_%d_%s_%s.txt" % (
                h, cutoff_type, args.mode.split('=')[0]))))
            
            for idx in range(len(fnames)-1,-1,-1):
                if not args.models is None and not fnames[idx].split('_')[0].split('/')[-1] in args.models:
                    del fnames[idx]
            
            if len(fnames) == 0:
                print ("H=%d files not found!" % h if args.mode == "gradnorms" else int(h/int(cutoff_type))); continue #exit()
            
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
                    plt.plot([0, len(values)-1], [values[0], values[-1]], color=plot_colors[idx], linestyle='--')
                    
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
                        print (model)
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
                    
                    try:
                        plt.plot(list(range(len(autocorrs[0]))), autocorrs[0], label=model, color=plot_colors[idx+1])
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
                plt.title("ACF averages of the self-attention-based models' test set forecasts and ground truth")
                plt.savefig("plots/autocorrs_%d_%s.png" % (h, "all_models" if args.models is None else "_".join(sorted(args.models))), dpi=600)
                #plt.show()
                plt.clf()

            if args.mode == "gradnorms":
                plt.legend([tuple(plots[model]) for model in plots], list(plots.keys()),
                                            handler_map={tuple: HandlerTuple(ndivide=None)})
        if args.mode == "gradnorms":
            plt.savefig("plots/gradnorms_%d_%s.png" % (h, "all_models" if args.models is None else "_".join(sorted(args.models))), dpi=600)
            #plt.show()
        
            # midpoints: 96, 192, 336, 720
            for k in midpts:
                if not k in midpts_plot:
                    midpts_plot[k] = [midpts[k][0][0]]
                else:
                    midpts_plot[k].append(midpts[k][0][0])
            
            for k in plots:
                midpts[k] = []
                plots[k] = []
            
            plt.clf()
            
            for idx, model in enumerate(poly_areas[cutoff_type].keys()):
                plt.plot([0, len(poly_areas[cutoff_type][model])], [0, 0])
                for cutoff_type in types:
                    plt.plot(list(range(len(poly_areas[cutoff_type][model]))), 
                            poly_areas[cutoff_type][model], label=model + "[%d->%d]" % (
                                1 if cutoff_type=="forward" else len(poly_areas[cutoff_type][model]),
                                len(poly_areas[cutoff_type][model]) if cutoff_type=="forward" else 1), 
                            color=plot_colors[idx], 
                            linestyle="dashed" if cutoff_type==types[1] else "dotted")

            plt.legend()
            plt.savefig("plots/gradnorms_%d_%s_areas.png" % (h, "all_models" if args.models is None else "_".join(sorted(args.models))), dpi=600)
            #plt.show()
            plt.clf()

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
    
    plt.savefig("plots/gradnorms_%d_%s_midpts.png" % (h, "all_models" if args.models is None else "_".join(sorted(args.models))), dpi=600)
    #plt.show()

