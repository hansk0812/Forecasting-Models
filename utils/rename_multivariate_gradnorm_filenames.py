import argparse
import os
import glob

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("logs_folder", help="Directory with <model>M<#> log files")
    ap.add_argument("dataset", help="<logs_folder>/<dataset>.txt is a text file with variable name mappings")
    ap.add_argument("model", help="Model name with multivariate gradnorms")

    args = ap.parse_args()
    
    variates = {}
    with open(os.path.join(args.logs_folder, args.dataset + ".txt"), 'r') as f:
        for l in f.readlines():
            m_key, variate = l.strip().split(' ')
            variates[m_key] = variate
    
    model_files_forward = glob.glob(os.path.join(args.logs_folder, args.model + "M*_forward_gradnorms.txt"))
    model_files_backward = [x.replace("forward", "backward") for x in model_files_forward]
    model_files = model_files_forward + model_files_backward

    for fname in model_files:
        key = fname.split(args.model)[-1].split('_')[0]
        model_name = args.model + ':' + variates[key]

        os.rename(fname, fname.replace("%s%s" % (args.model, key), model_name))
