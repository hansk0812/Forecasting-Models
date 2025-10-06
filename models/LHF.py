import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.SelfAttention_Family import FullAttention, ProbAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np

from . import FEDformer, Autoformer, Informer, Transformer
from . import DLinear, NLinear, TiDE, FiLM, CycleNet
from . import NBEATS, NHITS
from . import Pyraformer, Triformer
from . import xLSTM_TS
from . import NLinearLHF
from . import SpaceTime
from . import MultiResolutionDDPM

from utils.interp import interp1d

import gc

from model_size import model_size

import traceback
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    H/patches_size LHF models
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        self.patches_size = configs.patches_size
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        self.recurrent = configs.recurrent

        patch_model_configs = copy.deepcopy(configs)

        assert configs.pred_len % configs.patches_size == 0, "GUI assertion!"
        patch_model_configs.pred_len = configs.patches_size

        # Reduce network size to prevent overfitting on smaller horizon
        #configs.d_ff = configs.d_ff // 2
        #configs.d_model = math.floor(configs.d_model / 2)
        #configs.n_heads = configs.n_heads // 2

        self.networks = nn.ModuleList([self._build_model(patch_model_configs) for _ in range(self.pred_len//self.patches_size)])

        if not configs.load_from_chkpt is None:
            try:
                for net in self.networks:
                    chkpt_f = torch.load(configs.load_from_chkpt)
                    chkpt_m = net.state_dict()

                    for cf, cm in zip(chkpt_f, chkpt_m):
                        if not all([chkpt_f[cf].shape[idx]==chkpt_m[cm].shape[idx] for idx in range(len(chkpt_f[cf].shape))]):
                            print ("Skipping %s from checkpoint file!" % cf)
                            continue
                        #assert cf == cm, "Name conflict in checkpoint file: %s in file vs %s in model" % (cf, cm)
                        chkpt_m[cm] = chkpt_f[cf]

                print ("\n", "."*50, "\nLoaded %d copies of original model\n" % (self.pred_len // configs.patches_size), "."*50, "\n") 
            except Exception:
                traceback.print_exc()
                try:
                    self.load_state_dict(torch.load(configs.load_from_chkpt))
                    print ("\n", "."*50, "Loaded single checkpoint with all composing models!", "."*50, "\n") 
                except Exception:
                    traceback.print_exc()
                    print ("Cannot load checkpoint from file!")

        self.self_supervision = configs.self_supervised_patches

    def _build_model(self, args):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Pyraformer': Pyraformer,
            'Triformer': Triformer,
            'FiLM': FiLM,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'NHITS': NHITS,
            'NBEATS': NBEATS,
            'NLinearLHF': NLinearLHF,
            'TiDE': TiDE,
            'xLSTM_TS': xLSTM_TS,
            'CycleNet': CycleNet,
            'SpaceTime': SpaceTime,
            'MultiResolutionDDPM': MultiResolutionDDPM
        }
        model_name = args.model.split('/')[-1]
        model = model_dict[model_name].Model(args).float()

        if "LHF/" in args.model:
            try:
                if not args.load_from_chkpt is None:
                    model.load_state_dict(torch.load(args.load_from_chkpt, weights_only=True))
            except Exception:
                #traceback.print_exc()
                print ("COULDN'T LOAD CHECKPOINT INTO COMPOSED MODELS")
        else:
            if not args.load_from_chkpt is None:
                model.load_state_dict(torch.load(args.load_from_chkpt, weights_only=True))

        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)

        return model
    
    def split_dec(self, dec):
        num_models = self.pred_len // self.patches_size
        target_len = self.pred_len//num_models
        dec_labels = [dec[:,:self.label_len,:] for idx in range(num_models)]
        dec_preds = [dec[:,(self.label_len + idx*target_len):(self.label_len + (idx+1)*target_len),:] for idx in range(num_models)]
        outs = [torch.cat((x,y), dim=1) for x,y in zip(dec_labels, dec_preds)]
        return outs

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_dec_patch = self.split_dec(x_dec)
        x_mark_dec_patch = self.split_dec(x_mark_dec)

        if not dec_self_mask is None:
            raise NotImplementedError

        out_final, attns_final = [], []
        for idx in range(len(self.networks)):

            if not self.self_supervision is None and "io" in self.self_supervision:
                if idx > 0:
                    x_append = torch.cat((x_enc, out[:,-self.pred_len:,:]), dim=1)
                    if self.self_supervision == "io":
                        x_enc = x_append[:,self.pred_len:,:]
                    else: # io_interp
                        B = x_append.shape[0]
                        x_append = x_append.transpose(1, 2).view((-1, self.pred_len+self.seq_len))
                        x_enc = interp1d(torch.arange(0, self.pred_len+self.seq_len, 1),
                                         x_append,
                                         torch.arange(0, self.pred_len+self.seq_len, (self.seq_len+self.pred_len)/self.seq_len))

                    x_dec_patch[idx] = torch.cat((out[:,-self.label_len:,:], x_dec_patch[idx][:,self.label_len:,:]), dim=1)

            if self.output_attention:
                out, attns = self.networks[idx](x_enc, x_mark_enc, x_dec_patch[idx], x_mark_dec_patch[idx],
                                                enc_self_mask, dec_self_mask, dec_enc_mask)

                out_final.append(out[:, -self.pred_len:, :])
                attns_final.append(out)
            else:
                out = self.networks[idx](x_enc, x_mark_enc, x_dec_patch[idx], x_mark_dec_patch[idx],
                                            enc_self_mask, dec_self_mask, dec_enc_mask)
                out_final.append(out[:, -self.pred_len:, :])

        out_final = torch.concat(out_final, dim=1)
        if self.output_attention:
            attns_final = torch.concat(attns_final, dim=1)
            return out_final, attns_final
        else:
            return out_final  # [B, L, D]


if __name__ == '__main__':
    class Configs(object):
        ab = 0
        modes = 32
        mode_select = 'random'
        # version = 'Fourier'
        version = 'Wavelets'
        moving_avg = [12, 24]
        L = 1
        base = 'legendre'
        cross_activation = 'tanh'
        seq_len = 720
        label_len = 360
        pred_len = 720
        output_attention = False
        enc_in = 7
        dec_in = 7
        d_model = 512
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        
        factor = 22

        n_heads = 8
        d_ff = 2048
        e_layers = 1
        d_layers = 1
        c_out = 7
        activation = 'gelu'
        wavelet = 0
        
        load_from_chkpt = None
        use_multi_gpu = False
        use_gpu = True

        model = "FEDformer"
        patches_size = 720 #180

        # Informer, xLSTM
        distil = True
        
        # Triformer
        patch_sizes = (2,3,4)
    
        load_from_chkpt = None
           
        self_supervised_patches = None
        
    kwargs = {}
    kwargs["Autoformer"] = {"factor": 1, "d_model": 256}
    kwargs["Triformer"] = {"detail_freq": "(2,3,4)", "factor": 24}
    kwargs["Informer"] = {"factor": 1, "distil": True}
    kwargs["DLinear"] = {"factor": 7, "features": "M"}
    kwargs["NLinear"] = {"factor": 7, "features": "M"}
    kwargs["NLinearLHF"] = {"factor": 7, "features": "M", "start": 0.3, "step": 0.3, "lambdaval": 0.5}
    kwargs["TiDE"] = {"factor": 4, "features": "M"}
    kwargs["xLSTM_TS"] = {"factor": 24, "features": "M", "recurrent": False, "enc_in": 4, "e_layers": 1, "d_layers": 1, 
                            "d_model": 144, "n_heads": 6, "d_ff": 0, "modes": 1}
    kwargs["NHITS"] = {"enc_in": 1, "dec_in": 1, "features": "S"}
    kwargs["NBEATS"] = {"enc_in": 1, "dec_in": 1, "features": "S"}
    
    kwargs["Pyraformer"] = {"seq_len": 384, }

    SM_models = ["NBEATS", "NHITS"]

    for model_name in ['FEDformer', 'Autoformer', 'Informer', 'Triformer', 
                        'FiLM', 'DLinear', 'NLinear', 'NBEATS', 'NHITS', 
                        'NLinearLHF', 'TiDE', 'xLSTM_TS'][4:5]:
        
        print (model_name)
           
        configs = Configs()
        if model_name in kwargs:
            configs.__dict__.update(kwargs[model_name])
        
        configs.model = model_name
        
        import os, sys
        save_stdout = sys.stdout
        #sys.stdout = open('trash', 'w')
        
        model = Model(configs).to(device)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
        
        enc = torch.randn([3, configs.seq_len, 7 if not model_name in SM_models else 1]).to(device)
        enc_mark = torch.randn([3, configs.seq_len, 4]).to(device)
        dec = torch.randn([3, configs.seq_len//2+configs.pred_len, 7 if not model in SM_models else 1]).to(device)
        dec_mark = torch.randn([3, configs.seq_len//2+configs.pred_len, 4]).to(device)
        
        out = model.forward(enc, enc_mark, dec, dec_mark)
        
        sys.stdout = save_stdout
        #os.remove('trash')

        print(out.shape)
        print ("Model size:", model_size(model), "MB")

        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
