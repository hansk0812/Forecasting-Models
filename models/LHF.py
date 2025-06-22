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

from . import FEDformer, Autoformer, Informer, Transformer, Triformer
from . import DLinear, NLinear, TiDE, FiLM
from . import NBEATS, NHITS
from . import xLSTM_TS
from . import NLinearLHF


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    H/patch_length LHF models
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.patch_length = configs.patch_length
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        assert configs.pred_len % configs.patch_length == 0, "GUI assertion!"
        configs.pred_len = configs.patch_length
        self.networks = nn.ModuleList([self._build_model(configs) for _ in range(self.pred_len//self.patch_length)])

    def _build_model(self, args):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Triformer': Triformer,
            'FiLM': FiLM,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'NHITS': NHITS,
            'NBEATS': NBEATS,
            'NLinearLHF': NLinearLHF,
            'TiDE': TiDE,
            'xLSTM_TS': xLSTM_TS,
        }
        model_name = args.model.split('/')[-1]
        model = model_dict[model_name].Model(args).float()
        
        if not args.load_from_chkpt is None:
            model.load_state_dict(torch.load(args.load_from_chkpt, weights_only=True))

        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)

        return model

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        out_final, attns_final = [], []
        for idx in range(len(self.networks)):
            if self.output_attention:
                out, attns = self.networks[idx](x_enc, x_mark_enc, x_dec, x_mark_dec,
                                                enc_self_mask, dec_self_mask, dec_enc_mask)
                out_final.append(out[:, -self.pred_len:, :])
                attns_final.append(out)
            else:
                out = self.networks[idx](x_enc, x_mark_enc, x_dec, x_mark_dec,
                                            enc_self_mask, dec_self_mask, dec_enc_mask)
                print (out.shape)
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
        seq_len = 96
        label_len = 48
        pred_len = 96
        output_attention = False
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        
        # Triformer = 24, rest = 1
        factor = 24

        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        c_out = 7
        activation = 'gelu'
        wavelet = 0
        
        load_from_chkpt = None
        use_multi_gpu = False
        use_gpu = True

        model = "FEDformer"
        patch_length = 24

        # Informer
        distil = True
        
        # Triformer
        patch_sizes = (2,3,4)
    
    kwargs = {}
    kwargs["Autoformer"] = {"factor": 1}
    kwargs["Triformer"] = {"detail_freq": "(2,3,4)", "factor": 24}
    kwargs["Informer"] = {"factor": 1, "distil": True}
    kwargs["DLinear"] = {"factor": 7, "features": "M"}
    kwargs["NLinear"] = {"factor": 7, "features": "M"}
    kwargs["NLinearLHF"] = {"factor": 7, "features": "M", "start": 0.3, "step": 0.3, "lambdaval": 0.5}
    kwargs["TiDE"] = {"factor": 4, "features": "M"}
    kwargs["xLSTM_TS"] = {"factor": 24, "features": "M", "recurrent": False, "enc_in": 4, "d_model": 144, "n_heads": 4, "d_ff": 0, "modes": 2}
    kwargs["NHITS"] = {"enc_in": 1, "dec_in": 1, "features": "S"}
    kwargs["NBEATS"] = {"enc_in": 1, "dec_in": 1, "features": "S"}
    
    SM_models = ["NBEATS", "NHITS"]

    for model_name in ['FEDformer', 'Autoformer', 'Transformer', 'Informer',
                  'Triformer', 'FiLM', 'DLinear', 'NLinear', 'NBEATS', 'NHITS',
                  'NLinearLHF', 'TiDE', 'xLSTM_TS']:
        
        print (model_name)
        
        configs = Configs()
        if model_name in kwargs:
            configs.__dict__.update(kwargs[model_name])
        
        configs.model = model_name
        
        import sys
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        
        model = Model(configs).to(device)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
        
        enc = torch.randn([3, configs.seq_len, 7 if not model_name in SM_models else 1]).to(device)
        enc_mark = torch.randn([3, configs.seq_len, 4]).to(device)
        dec = torch.randn([3, configs.seq_len//2+configs.pred_len, 7 if not model in SM_models else 1]).to(device)
        dec_mark = torch.randn([3, configs.seq_len//2+configs.pred_len, 4]).to(device)
        
        out = model.forward(enc, enc_mark, dec, dec_mark)
        sys.stdout = save_stdout
        print(out.shape)
