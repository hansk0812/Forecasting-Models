import argparse

import os
import glob
import json

import sys
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

def remove_param(setting, param_shorthand):
    
    setting_split = setting.split(param_shorthand)
    setting = setting_split[0] + '_'.join(setting_split[1].split('_')[1:])

    return setting

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--model', type=str, default='FEDformer',
                        help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

    # supplementary config for FEDformer model
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--detail_freq', type=str, default='h', help='like freq, but use in predict')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str, default='tanh'

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=4, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')
    
    parser.add_argument('--load_from_chkpt', default=None, help="Path to pretrained model to resume training from")
    parser.add_argument('--gpu_memory_usage', action="store_true", help="If True, prints GPU memory usage summary and exits")
    parser.add_argument('--inspect_backward_pass', default=None, help="Uses 0-masked loss [forward, backward] to inspect gradients by horizon")
    parser.add_argument('--calculate_acf', default=None, type=int, help="Uses a lag of value specified as argument for ACF")

    parser.add_argument('--model_params_json', default=None, help="Path to JSON file with model hyperparameters and model zoo dir if available")
    parser.add_argument('--patches_size', default=None, type=int, help="Divide H into H/patches_size models")
    parser.add_argument('--self_supervised_patches', type=str, default=None, help="""Add self-supervision to patches-based splitting of models:
                            [io, io_interp]""")
    
    parser.add_argument('--start', default=1, type=float, help="AR SS arange param1")
    parser.add_argument('--step', default=1, type=float, help="AR SS arange param2")
    parser.add_argument('--lambdaval', default=0.5, type=float, help="AR SS weightage param")
    
    parser.add_argument('--recurrent', action='store_true', help="xLSTM recurrence flag")
    
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    if not args.model_params_json is None and os.path.exists(args.model_params_json):
        with open(args.model_params_json, 'r') as f:
            params = json.load(f)
        
        if "LHF/" in args.model:
            single_model = args.model.split("/")[-1]
        else:
            single_model = args.model
        
        json_ft = args.features if args.features != "SM" else "M"
        model_params = params["models"][json_ft][single_model][str(args.pred_len)]
        
        try:
            chkpt_path = glob.glob(os.path.join(params["zoo_path"], single_model, args.features, 
                                    "*sl%d_*pl%d*" % (model_params["seq_len"], model_params["pred_len"])))[0]

            if os.path.isdir(chkpt_path):
                chkpt_path = os.path.join(chkpt_path, "checkpoint.pth")
            
            args.load_from_chkpt = chkpt_path

            if args.patches_size==0 and args.is_training:
                print ("Using model from model zoo with the same metric values")
                exit()
    
        except Exception:
            
            import traceback
            traceback.print_exc()
            print ("\n\n", "."*75, "\n")
            print ("\t Model: %s ; Horizon: %d CHECKPOINT FILE NOT FOUND IN ZOO" % (args.model, args.pred_len))
            print ("\n\n", "."*75, "\n")

        for param in model_params:
            setattr(args, param, model_params[param])

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_pt{}_ss{}_eb{}_dt{}_{}_{}'.format(
                args.task_id,
                args.model,
                args.mode_select,
                args.modes,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.patches_size,
                args.self_supervised_patches,
                args.embed,
                args.distil,
                args.des,
                ii)
            if args.patches_size == 0 or args.patches_size is None:
                setting = remove_param(setting, "pt")
            if args.self_supervised_patches is None:
                setting = remove_param(setting, "ss")

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_pt{}_ss{}_eb{}_dt{}_{}_{}'.format(
                args.task_id,
                args.model,
                args.mode_select,
                args.modes,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.patches_size,
                args.self_supervised_patches,
                args.embed,
                args.distil,
                args.des,
                ii)
        if args.patches_size == 0 or args.patches_size is None:
            setting = remove_param(setting, "pt")
        if not args.self_supervised_patches:
            setting = remove_param(setting, "ss")

        if not args.model_params_json is None:
            chkpt_symlink = os.path.join("checkpoints", setting, "checkpoint.pth")
            if not os.path.exists(os.path.dirname(chkpt_symlink)):
                os.makedirs(os.path.dirname(chkpt_symlink))

            if not args.load_from_chkpt is None and not os.path.islink(chkpt_symlink):
                if not os.path.exists(chkpt_symlink):
                    os.symlink(args.load_from_chkpt, chkpt_symlink)

        exp = Exp(args)  # set experiments
        print ('test > mse:', args.model, args.pred_len, 'horizon size')
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
