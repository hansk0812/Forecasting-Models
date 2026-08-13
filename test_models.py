import glob
import os

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("dataset", help="Name of dataset")
ap.add_argument("root_path", help="Dataset directory")
ap.add_argument("data_path", help="Dataset file")
ap.add_argument("model", help="Name of model")
ap.add_argument("horizon", help="Size of horizon", type=int)
ap.add_argument("num_variates", help="Number of variates", type=int)
ap.add_argument("--args", default="", help="Additional arguments to script")
args = ap.parse_args()

files = glob.glob("checkpoints/*%s*_pl%d_*/checkpoint.pth" % (args.model, args.horizon))

for fl in files:
    fname = fl.split('/')[1]
    vals = fname.split('_')

    task_id = vals[0]
    model = vals[1]
    mode_select = vals[2]
    modes = vals[3][5:]
    data = vals[4]
    features = vals[5][2:]
    seq_len = vals[6][2:]
    label_len = vals[7][2:]
    pred_len = vals[8][2:]
    d_model = vals[9][2:]
    n_heads = vals[10][2:]
    e_layers = vals[11][2:]
    d_layers = vals[12][2:]
    d_ff = vals[13][2:]
    factor = vals[14][2:]
    embed = vals[15][2:]
    distil = "" if bool(vals[16][2:]) else "--distil"
    des = vals[17]
    
    run = "python run.py --load_from %s --data %s --root_path %s --data_path %s --is_train 0 --task_id %s --model %s --mode_select %s --modes %s --data %s --features %s --seq_len %s --label_len %s --pred_len %s --d_model %s --n_heads %s --e_layers %s --d_layers %s --d_ff %s --factor %s --embed %s --distil %s --des %s --enc_in %d --dec_in %d --c_out %d %s" % (fl, args.dataset, args.root_path, args.data_path, task_id, model, mode_select, modes, data, features, seq_len, label_len, pred_len, d_model, n_heads, e_layers, d_layers, d_ff, factor, embed, distil, des, args.num_variates, args.num_variates, args.num_variates, args.args)
    
    print (run)
    os.system(run)
