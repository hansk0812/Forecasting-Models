import os
import glob

import h5py

import time
import warnings
import numpy as np

import traceback

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

from models import FEDformer, Autoformer, Informer, Transformer
from models import DLinear, NLinear, NHITS, TiDE, NBEATS, FiLM
from models import Pyraformer, Triformer
from models import xLSTM_TS
from models import SpaceTime
from models import MultiResolutionDDPM

from models import SAMformer
from models import CycleNet

from models import NLinearLHF
from models import LHF

from models import AutoformerSansRoll
from models import NLinearSansNorm

from models import PyraformerSansMask, PyraformerOppMask, PyraformerEncoderOppMask

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

from utils.metrics import WeatherMetricsCalculator

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
cmap = colormaps["Reds"]

from matplotlib.animation import FuncAnimation
from functools import partial

np.random.seed(1)

from scipy.signal import correlate
from statsmodels.tsa.stattools import acf

from tqdm import tqdm

from decimal import Decimal

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import random
import time
seed_everything(int(time.time()))

#TODO: Add other metrics
class RunningAvgMetrics:

    sum_val1 = 0
    sum_val2 = 0
    count = 0

    def __init__(self, shape, metric):
        
        if metric == "mse":
            self.metric_fn = self.mse
        elif metric == "mae":
            self.metric_fn = self.mae
        else:
            raise NotImplementedError
        
        if len(shape) == 1 and shape[0] == 1:
            return
        
        self.metric_sum = np.zeros(shape)
    
    def mse(self, y_p, y_g):
        return ((y_p - y_g)**2).sum(axis=0)

    def mae(self, y_p, y_g):
        return np.abs(y_p - y_g).sum(axis=0)
    
    def add_to_mean(self, y_pred, y_gt):
        self.count += y_gt.shape[0]
        self.metric_sum += self.metric_fn(y_pred, y_gt)

    def get_mean(self):
        return self.metric_sum / float(self.count)

class BackwardPassInspectLoss(nn.Module):

    def __init__(self, horizon, cutoff, cutoff_type, device, loss="mae", return_batch_dim=False, multivariate_softmax=False):
        # cutoff_type=forward: For a given horizon, 0:cutoff are 1 and cutoff:horizon are 0
        # cutoff_type=backward: For a given horizon, 0:cutoff are 0 and cutoff:horizon are 1
        # return_batch_dim=True for DDPM model
        
        super().__init__()
        assert cutoff <= horizon, "GUI Assertion!"

        self.mask = torch.ones(horizon).to(device)
        if cutoff_type == "forward":
            self.mask[cutoff:] = 0 # pytorch automatically ignores indices outside the range of the tensor's shape!
        else:
            self.mask[:cutoff] = 0

        if loss.lower() == "mse":
            self.loss_fn = self.MSE_per_timestep
        else:
            self.loss_fn = self.MAE_per_timestep

        self.mean_dims_per_timestep = (0,2) if not return_batch_dim else (2,)
        self.mean_dims_return = (0,) if not return_batch_dim else (1,)

        if multivariate_softmax:
            self.mean_dims_per_timestep = tuple(x for x in self.mean_dims_per_timestep if x != 2)
            self.mask.unsqueeze_(1)

    def MSE_per_timestep(self, x, y):
        # B, H, D
        if len(self.mean_dims_per_timestep) > 0:
            return torch.mean((x-y)**2, dim=self.mean_dims_per_timestep)
        else:
            return (x-y)**2

    def MAE_per_timestep(self, x, y):
        # B, H, D
        if len(self.mean_dims_per_timestep) > 0:
            return torch.mean((x-y)**2, dim=self.mean_dims_per_timestep)
        else:
            return torch.abs(x-y)

    def forward(self, x, y):
        
        loss = self.loss_fn(x, y)
        loss *= self.mask
        return torch.mean(loss, dim=self.mean_dims_return)

import numpy as np

def matplotlib_animation(input_gradnorms, batch_x, batch_y, ham_gradnorms, fname):
    # input_gradnorms: (H+1, L, V), one over each causal and anti-causal mask from 0 to H

    assert len(fname.split('.')) == 2 and fname[-3:] == "mp4"

    #colours = list(matplotlib.colors.cnames.values())[:input_gradnorms.shape[-1]]
    colours = plt.cm.get_cmap("tab20b", input_gradnorms.shape[-1])
    for v_idx in range(input_gradnorms.shape[-1]):
        
        fname_format = fname.split('.')
        v_fname = fname_format[0] + "_variate%d." % v_idx + fname_format[1]

        fig, ax = plt.subplots(5, sharex=True, height_ratios=[2,1,1,2,1], figsize=(16,9))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        x_axis = np.arange(1, input_gradnorms.shape[-2] + 1, 1)
        extent = [x_axis[0]-(x_axis[1]-x_axis[0])/2., x_axis[-1]+(x_axis[1]-x_axis[0])/2.,0,1]
        
        for v_idx_plot in range(input_gradnorms.shape[-1]):
            ax[0].plot(x_axis, batch_x[:, v_idx_plot] / batch_x[:, v_idx_plot].max(), 
                       color=colours(v_idx_plot)) #colours[v_idx_plot]) 
            ax[0].set_ylabel("X")
        
        gradient_extents = [input_gradnorms[:,:,v_idx].min(axis=(0,1)), input_gradnorms[:,:,v_idx].max(axis=(0,1))]
        ax[2].set_ylim(gradient_extents)
        heatmap_artist = ax[1].imshow(input_gradnorms[0, :, v_idx][np.newaxis, :], 
                                      cmap="bwr", extent=extent, aspect="auto", 
                                      vmin = gradient_extents[0], vmax = gradient_extents[1])
        plot_artist, = ax[2].plot(x_axis, input_gradnorms[0, :, v_idx])
        ax[2].set_ylabel("(dL_dX)t")

        for v_idx_plot in range(input_gradnorms.shape[-1]):
            ax[3].plot(x_axis, batch_y[:, v_idx_plot] / batch_y[:, v_idx_plot].max(), 
                       color=colours(v_idx_plot)) #colours[v_idx_plot])
            
        points_artist, = ax[3].plot([], [], 'o', color=colours(v_idx)) #colours[v_idx])
        ax[3].set_ylabel("Y")

        gradnorms_x_axis = np.arange(0, ham_gradnorms.shape[0])
        ax[4].plot(gradnorms_x_axis, ham_gradnorms)
        ax[4].set_ylabel("(dL_dW)B")
        points_artist2, = ax[4].plot([], [], 'o', color=colours(v_idx)) #colours[v_idx])

        def animate(idx, v_idx = None):

            heatmap_artist.set_data(input_gradnorms[idx, :, v_idx][np.newaxis,:])
            plot_artist.set_xdata(x_axis)
            plot_artist.set_ydata(input_gradnorms[idx, :, v_idx])
            
            points_artist.set_xdata(x_axis[idx: idx + 1])
            points_artist.set_ydata(batch_y[idx: idx + 1, v_idx] / batch_y[:, v_idx].max())
            
            points_artist2.set_xdata(gradnorms_x_axis[idx: idx + 1])
            points_artist2.set_ydata(ham_gradnorms[idx: idx + 1])

            return heatmap_artist, plot_artist, points_artist
     
        animation = FuncAnimation(fig, partial(animate, v_idx=v_idx), frames=range(1, input_gradnorms.shape[0]), init_func=None, blit=True, interval=30)
        animation.save(v_fname, writer="ffmpeg")

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):

        if "LHF/" in self.args.model:
            model = LHF.Model(self.args).float()    
        else:
            model_dict = {
                'FEDformer': FEDformer,
                'Autoformer': Autoformer,
                'Transformer': Transformer,
                'Informer': Informer,
                'Triformer': Triformer,
                'FiLM': FiLM,
                'DLinear': DLinear,
                'NLinear': NLinear,
                'NLinearLHF': NLinearLHF,
                'NHITS': NHITS,
                'TiDE': TiDE,
                'NBEATS': NBEATS,
                'Pyraformer': Pyraformer,
                'SAMformer': SAMformer,
                'CycleNet': CycleNet,
                'SpaceTime': SpaceTime,
                'MultiResolutionDDPM': MultiResolutionDDPM,
                'AutoformerSansRoll': AutoformerSansRoll,

                'NLinearSansNorm': NLinearSansNorm,

                'PyraformerSansMask': PyraformerSansMask,
                'PyraformerOppMask': PyraformerOppMask,
                'PyraformerEncoderOppMask': PyraformerEncoderOppMask,            
            }
            try:
                model_dict['xLSTM_TS'] = xLSTM_TS
                if self.args.model == 'xLSTM_TS':
                    import xlstm
                    xlstm_dir = os.path.dirname(xlstm.__file__)
                    os.system(
                            "sed -i \"s/self.config.embedding_dim=.*/self.config.embedding_dim=%d/\" \"%s/blocks/slstm/layer.py\"" \
                                    % (self.args.d_model, xlstm_dir))
                    os.system(
                            "sed -i \"s/self.config.embedding_dim = .*/self.config.embedding_dim = %d/\" \"%s/blocks/mlstm/layer.py\"" \
                                    % (self.args.d_model, xlstm_dir))
                    os.system(
                            "sed -i \"s/embedding_dim: int = .*/embedding_dim: int = %d/\" %s/xlstm_block_stack.py" \
                                    % (self.args.d_model, xlstm_dir))
                    
                    print ("xLSTM import complete with changes to package!")

            except Exception:
                print ("sed ERROR!")
                import traceback
                traceback.print_exc()
                pass

            model = model_dict[self.args.model].Model(self.args).float()
            
        if not self.args.load_from_chkpt is None:
            if "LHF/" in self.args.model:
                try:
                    model.load_state_dict(torch.load(self.args.load_from_chkpt, weights_only=True))
                except Exception:
                    #import traceback
                    #traceback.print_exc()
                    print ("COULDN'T LOAD CHECKPOINT FROM FILE OVER PATCHES MODELS! 1 vs n NETWORKS, SIZE DIFFERENCES")
            else:
                try:
                    #1/0
                    model.load_state_dict(torch.load(self.args.load_from_chkpt, weights_only=True))
                    print ("\n", "."*50, "\n\nLoaded initial model from %s\n\n" % self.args.load_from_chkpt, "."*50)
                except Exception:
                    try:
                        # 1. Load the state dict
                        state_dict = torch.load(self.args.load_from_chkpt, weights_only=True, map_location="cpu")

                        # 2. Strip the 'module.' prefix from keys
                        new_state_dict = OrderedDict()
                        for k, v in state_dict.items():
                            name = k if k.startswith("module.") else "module." + k
                            new_state_dict[name] = v                        
                        
                        model.load_state_dict(new_state_dict)
                        print ("Loaded single GPU model onto DataParallel model")
                    except Exception:
                        pass
           
        # fft doesn't work well with DataParallel, switching to DDP
        #if self.args.use_multi_gpu and self.args.use_gpu:
        #    model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, backward_pass_inspect_cutoff=None, inspect_type=None, horizon=None, multivariate_softmax=None):
        if backward_pass_inspect_cutoff is None:
            criterion = nn.MSELoss()
        else:
            assert not horizon is None, "Interpreters!"
            criterion = BackwardPassInspectLoss(horizon, backward_pass_inspect_cutoff, 
                                                inspect_type, device=self.device, loss=self.args.loss, multivariate_softmax=multivariate_softmax)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                
                # Initial test set loss calculation needs to consider weather dataset's test set
                if isinstance(batch_x, (list, tuple)):
                    batch_x, _ = batch_x

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                if self.args.model == "MultiResolutionDDPM":
                    batch_y = batch_y.to(self.device)
                
                if not self.args.model == "SpaceTime":
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    if not self.args.model == "MultiResolutionDDPM":
                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        
                        if self.args.model == "CycleNet":
                            outputs = self.model(batch_x, batch_cycle)
                        elif self.args.model == "SpaceTime":
                            (outputs, _), _ = self.model(batch_x)
                        elif self.args.model == "MultiResolutionDDPM":
                            loss = self.model.train_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == "CycleNet":
                        outputs = self.model(batch_x, batch_cycle)
                    elif self.args.model == "SpaceTime":
                        (outputs, _), _ = self.model(batch_x)
                    elif self.args.model == "MultiResolutionDDPM":
                        loss = self.model.train_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                if self.args.model != "MultiResolutionDDPM":
                   loss = criterion(outputs, batch_y)

                total_loss.append(loss)
        total_loss = torch.mean(torch.stack(total_loss))
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
 
        if not self.args.inspect_backward_pass is None and "_batchsize" in self.args.inspect_backward_pass:
            batchsize_timestep = int(self.args.inspect_backward_pass.split('=')[-1])
            self.args.inspect_backward_pass = self.args.inspect_backward_pass.split("_batchsize")[0]
            skip_zeroes = True
        else:
            skip_zeroes = False
        
        if not self.args.inspect_backward_pass is None:
            if self.args.backward_pass_set == "val":
                train_data, train_loader = vali_data, vali_loader
            elif self.args.backward_pass_set == "test":
                train_data, train_loader = test_data, test_loader

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if not self.args.load_from_chkpt is None:
            try:
                state_dict = torch.load(self.args.load_from_chkpt)
            except Exception:
                state_dict = torch.load(self.args.load_from_chkpt, map_location="cpu")

            self.model.load_state_dict(state_dict)

        import time
        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        test_loss = self.vali(test_data, test_loader, criterion)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, val_loss_min=test_loss)
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        batch_start = 0
        if not self.args.inspect_backward_pass is None:

            # Per layer norms
            if '=' in self.args.inspect_backward_pass and not "batchsizes" in self.args.inspect_backward_pass:
                layer_name = self.args.inspect_backward_pass.split('=')[-1]
                if layer_name[0] != layer_name[-1]:
                    raise AssertionError("Use parantheses for layer names")
                layer_names = [x.strip() for x in layer_name[1:-1].split(',')]
            else:
                layer_names = [n[0] for n in self.model.named_parameters()]

            if not os.path.isdir(self.args.gradnorms_dir):
                os.makedirs(self.args.gradnorms_dir)

            input_grads_dir = os.path.join(self.args.gradnorms_dir, "%s_%d_%s_input_gradnorms" % (
                                                self.args.model, self.args.pred_len, self.args.inspect_backward_pass))
            if not os.path.isdir(input_grads_dir):
                os.makedirs(input_grads_dir)

            if not self.args.backward_pass_multivariate:
                input_gradnorms_shape = (self.args.pred_len + 1, 
                                         self.args.batch_size, 
                                         self.args.seq_len, 
                                         self.args.enc_in)
            else:
                input_gradnorms_shape = (self.args.pred_len + 1,
                                         self.args.enc_in,
                                         self.args.batch_size, 
                                         self.args.seq_len, 
                                         self.args.enc_in)

            gradnorms_file = os.path.join(self.args.gradnorms_dir, "%s_%s_%d_%s.pth" % (
                                                      self.args.data, self.args.model, 
                                                      self.args.pred_len, self.args.inspect_backward_pass)
            if os.path.exists(gradnorms_file):
                try:
                    load_dict = torch.load(gradnorms_file)
                    grad_norms_per_timestep = load_dict["gradnorms"]
                    batch_start = load_dict["batch"] + 1

                    if batch_start == len(train_loader):
                        exit()
                except Exception:
                    if not self.args.backward_pass_multivariate:
                        grad_norms_per_timestep = {"forward": [torch.zeros((len(train_loader), len(layer_names))) \
                                                                    for _ in range(self.args.pred_len+1)],
                                                   "backward": [torch.zeros((len(train_loader), len(layer_names))) \
                                                                        for _ in range(self.args.pred_len+1)]}

                    else:
                        #TODO: Data-based splits for SM models
                        grad_norms_per_timestep = {"forward": [torch.zeros((len(train_loader), len(layer_names), self.args.enc_in)) \
                                                                    for _ in range(self.args.pred_len+1)],
                                                   "backward": [torch.zeros((len(train_loader), len(layer_names), self.args.enc_in)) \
                                                                        for _ in range(self.args.pred_len+1)]}
 
            else:
                if not self.args.backward_pass_multivariate:
                    grad_norms_per_timestep = {"forward": [torch.zeros((len(train_loader), len(layer_names))) \
                                                                for _ in range(self.args.pred_len+1)],
                                               "backward": [torch.zeros((len(train_loader), len(layer_names))) \
                                                                for _ in range(self.args.pred_len+1)]}
                    
                else:
                    #TODO: Data-based splits for SM models
                    grad_norms_per_timestep = {"forward": [torch.zeros((len(train_loader), len(layer_names), self.args.enc_in)) \
                                                                for _ in range(self.args.pred_len+1)],
                                               "backward": [torch.zeros((len(train_loader), len(layer_names), self.args.enc_in)) \
                                                                    for _ in range(self.args.pred_len+1)]}
                    
        elif not self.args.calculate_acf is None:
            autocorrs = []
       
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()
            train_loader = tqdm(train_loader, leave=False)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
               
                if i < batch_start:
                    continue

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                if not self.args.inspect_backward_pass is None:
                    batch_x.requires_grad_()
                    batch_x_mark.requires_grad_()
                    batch_y_mark.requires_grad_()

                if not self.args.model == "SpaceTime":
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    if not self.args.model == "MultiResolutionDDPM":
                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == "CycleNet":
                            outputs = self.model(batch_x, batch_cycle)
                        elif self.args.model == "SpaceTime":
                            (outputs, y_o), (z_p, z_g) = self.model(batch_x)
                        elif self.args.model == "MultiResolutionDDPM":
                            loss = self.model.train_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)
                        else:
                             if self.args.output_attention:
                                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                             else:
                                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        if self.args.model != "MultiResolutionDDPM":
                            batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                            loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model == "CycleNet":
                        outputs = self.model(batch_x, batch_cycle)
                    elif self.args.model == "SpaceTime":
                        (outputs, y_o), (z_p, z_g) = self.model(batch_x)
                    elif self.args.model == "MultiResolutionDDPM":
                        loss = self.model.train_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    if self.args.model != "MultiResolutionDDPM":
                        batch_y = batch_y[:, -self.args.pred_len:, :]
                        loss = criterion(outputs, batch_y)

                    if self.args.model == "SpaceTime":
                        xy = torch.cat((batch_x, batch_y), dim=1)
                        loss += criterion(y_o[:, self.model.kernel_dim-1:, :], xy[:, self.model.kernel_dim:self.args.seq_len+1, :])
                        loss += criterion(z_p[:, self.model.kernel_dim-1:-1, :], z_g[:, self.model.kernel_dim:, :])
                    
                    if self.args.model != "MultiResolutionDDPM" or (
                        self.args.inspect_backward_pass is None and self.args.model == "MultiResolutionDDPM"):
                        
                        if not self.args.inspect_backward_pass is None and self.args.backward_pass_multivariate:
                            if len(loss.shape) > 0:
                                loss = loss[0]

                        train_loss.append(loss.item())
                
                    if self.args.model == "MultiResolutionDDPM" and not self.args.inspect_backward_pass is None:
                        outputs = torch.concatenate([x[0] for x in loss], dim=0)
                        batch_y = torch.concatenate([x[1] for x in loss], dim=0)

                if i==5 and self.args.gpu_memory_usage:
                    from model_size import model_size
                    print ("MEMORY: Model Size: %s: %fMB" % (self.args.model, model_size(self.model)))
                    print ("MEMORY: GPU summary (in GB) after backward pass:")
                    print ("allocated per data pt:", torch.cuda.memory_allocated(self.device)/(1024.*1024*1024*self.args.batch_size))
                    print ("reserved per data pt:", torch.cuda.memory_reserved(self.device)/(1024.*1024*1024*self.args.batch_size))
                    print ("Time per epoch: %f hours" % ((time.time()-epoch_time)*len(train_loader)/(6.*60*60)))
                    exit()               

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if self.args.inspect_backward_pass is None and not self.args.calculate_acf:
                        loss.backward()
                        model_optim.step()
                    elif not self.args.calculate_acf is None:
                        if epoch > 0:
                            #print ([len(x) for x in autocorrs])
                            print ("Autocorrelation for %s pred:" % self.args.model, np.array(autocorrs)[:,:,1:2,:].mean(axis=(0,1,2)))
                            print ("Autocorrelation for %s gt:" % self.args.model, np.array(autocorrs)[:,:,0:1,:].mean(axis=(0,1,2)))
                            exit()
                        
                        for b in range(batch_y.shape[0]):
                            feature_autocorrs = []
                            for f in range(batch_y.shape[2]):
                                #autocorrs.append([
                                #    correlate(batch_y[b,:,f].detach().cpu().numpy(), batch_y[b,:,f].detach().cpu().numpy(), method="fft"),
                                #    correlate(outputs[b,:,f].detach().cpu().numpy(), outputs[b,:,f].detach().cpu().numpy(), method="fft")])
                                
                                autocorr_pred = acf(batch_y[b,:,f].detach().cpu().numpy(), nlags=self.args.calculate_acf)
                                autocorr_gt = acf(outputs[b,:,f].detach().cpu().numpy(), nlags=self.args.calculate_acf)
                                
                                if not (np.any(np.isnan(autocorr_pred)) or np.any(np.isnan(autocorr_gt))):
                                    feature_autocorrs.append([autocorr_pred, autocorr_gt])
                                else:
                                    break
                                
                                if f == batch_y.shape[2]-1:
                                    autocorrs.append(feature_autocorrs)

                    else:

                        if any([x in self.args.model for x in ["former", "SpaceTime", "DDPM"]]) and not skip_zeroes:
                            step = 50
                        else:
                            step = 1

                        if epoch > 0:

                            if not os.path.isdir(os.path.dirname(gradnorms_file)):
                                os.path.makedirs(os.path.dirname(gradnorms_file))

                            gradnorms_txt = os.path.join(self.args.gradnorms_dir, 
                                                         "%s_%s_%d_%s_gradnorms.txt" % (
                                                            self.args.model,
                                                            self.args.data,
                                                            self.args.pred_len,
                                                            self.args.inspect_backward_pass))
                            if not self.args.backward_pass_multivariate:
                                # output gradients
                                with open(gradnorms_txt, "a") as f:
                                    for idx in range(0, self.args.pred_len+1, step):

                                        if skip_zeroes:
                                            if (self.args.inspect_backward_pass == "forward" and idx!=batchsize_timestep):
                                                continue
                                    
                                        norms_str = ' '.join(["%s=%.16E" % (n, Decimal(g.item())) for n, g in \
                                                                zip(layer_names, 
                                                                    grad_norms_per_timestep[
                                                                        self.args.inspect_backward_pass][idx].mean(axis=0))])
                                        if self.args.inspect_backward_pass == "backward": # 0:idx entries are 0
                                            print ("Grad norm for H: %d->%d: %s" % (idx, self.args.pred_len, norms_str), file=f)
                                        else:
                                            print ("Grad norm for H: %d->%d: %s" % (0, idx, norms_str), file=f)
                                
                            else:
                                multivariate_gradnorms_dir = gradnorms_txt.replace("gradnorms.txt", "gradnorms_M")
                                if not os.path.isdir(multivariate_gradnorms_dir):
                                    os.makedirs(multivariate_gradnorms_dir)

                                # Added multivariate softmax HAM plots
                                for v_idx in range(self.args.enc_in):
                                    
                                    with open(os.path.join(multivariate_gradnorms_dir, 
                                                "%s_%s_%d_V%d_forward_gradnorms.txt" % (
                                                self.args.model, self.args.data, 
                                                self.args.pred_len, v_idx)), 'w') as f:
                                        with open(os.path.join(multivariate_gradnorms_dir, 
                                                  "%s_%s_%d_V%d_backward_gradnorms.txt" % (
                                                  self.args.model, self.args.data,
                                                  self.args.pred_len, v_idx)), 'w') as g:
                                            
                                            for idx in range(0, self.args.pred_len+1, step):

                                                if skip_zeroes:
                                                    if (self.args.inspect_backward_pass == "forward" and idx!=batchsize_timestep):
                                                        continue

                                                if self.args.inspect_backward_pass == "backward": # 0:idx entries are 0
                                                    norms_str = ' '.join(["%s=%.16E" % (n, Decimal(g.item())) for n, g in \
                                                                        zip(layer_names, grad_norms_per_timestep["backward"][idx].mean(axis=0)[:, v_idx])])
                                                    g.write("Grad norm for H: %d->%d: %s\n" % (idx, self.args.pred_len, norms_str))
                                                else:
                                                    norms_str = ' '.join(["%s=%.16E" % (n, Decimal(g.item())) for n, g in \
                                                                        zip(layer_names, grad_norms_per_timestep["forward"][idx].mean(axis=0)[:, v_idx])])
                                                    f.write("Grad norm for H: %d->%d: %s\n" % (0, idx, norms_str))
                                        
                            exit()

                        if self.args.model in ["Informer", "Autoformer", "FEDformer", "Pyraformer", "Triformer"]:
                            input_grad_norms = [torch.zeros(input_gradnorms_shape),
                                                torch.zeros(input_gradnorms_shape[:-1] + (5,)),
                                                torch.zeros(input_gradnorms_shape[:-1] + (5,))]
                        else:
                            input_grad_norms = [torch.zeros(input_gradnorms_shape)]

                        for h in range(0, self.args.pred_len+1, step):

                            if skip_zeroes:
                                if (self.args.inspect_backward_pass == "forward" and h!=batchsize_timestep):
                                    continue

                            criterion = self._select_criterion(backward_pass_inspect_cutoff=h, 
                                            inspect_type=self.args.inspect_backward_pass, horizon=self.args.pred_len,
                                            multivariate_softmax=self.args.backward_pass_multivariate)
                            loss = criterion(outputs, batch_y)
                            
                            if not self.args.backward_pass_multivariate:
                                
                                loss.backward(retain_graph=True)
                            
                                for idx, (n, param) in enumerate(self.model.named_parameters()):
                                    if not n in layer_names:
                                        print ("CONTINUE")
                                        continue
                                    if not param.grad is None:
                                        #grad_norms.append(param.grad.norm())
                                        grad_norms_per_timestep[self.args.inspect_backward_pass][h][i][idx] = param.grad.norm()
                                 
                                # input gradients
                                input_grad_norms[0][h] = batch_x.grad.cpu()
                                if len(input_grad_norms) > 1:
                                    input_grad_norms[1][h][..., :batch_x_mark.shape[-1]] = batch_x_mark.grad.cpu()
                                    input_grad_norms[2][h][..., :batch_y_mark.shape[-1]] = batch_y_mark.grad.cpu()

                                for param in self.model.parameters():
                                    if not param.grad is None:
                                        param.grad.fill_(0)
                                batch_x.grad.fill_(0)
                                if len(input_grad_norms) > 1:
                                    batch_x_mark.grad.fill_(0)
                                    batch_y_mark.grad.fill_(0)
                            
                            else:

                                for v_idx in range(self.args.enc_in):
                                    v_loss = loss[...,v_idx:v_idx+1].mean()
                                    v_loss.backward(retain_graph=True)

                                    for idx, (n, param) in enumerate(self.model.named_parameters()):
                                        if not n in layer_names:
                                            print ("CONTINUE")
                                            continue
                                        if not param.grad is None:
                                            #grad_norms.append(param.grad.norm())
                                            grad_norms_per_timestep[self.args.inspect_backward_pass][h][i][idx][v_idx] = param.grad.norm()
                                    
                                    # input gradients
                                    input_grad_norms[0][h, v_idx] = batch_x.grad.cpu()
                                    if len(input_grad_norms) > 1:
                                        input_grad_norms[1][h, v_idx] = batch_x_mark.grad.cpu()
                                        input_grad_norms[2][h, v_idx] = batch_y_mark.grad.cpu()

                                    for param in self.model.parameters():
                                        if not param.grad is None:
                                            param.grad.fill_(0)
                                    batch_x.grad.fill_(0)
                                    if len(input_grad_norms) > 1:
                                        batch_x_mark.grad.fill_(0)
                                        batch_y_mark.grad.fill_(0)
                                
                                loss = v_loss
                        
                        for batch_idx in range(input_grad_norms[0].shape[-3]):
                            data_idx = self.args.batch_size * i + batch_idx
                            matplotlib_animation(input_grad_norms[0][:, batch_idx, :, :].numpy(), batch_x.detach().cpu().numpy()[batch_idx], 
                                                 batch_y.detach().cpu().numpy()[batch_idx], 
                                                 torch.stack(grad_norms_per_timestep[self.args.inspect_backward_pass]).numpy()[:, i, :].mean(axis=-1), 
                                                 fname = os.path.join(input_grads_dir, "data_id%d.mp4" % data_idx))
                        #torch.save(input_grad_norms[0], input_grads_file.split('.')[0] + '.pth') 
                        #with h5py.File(input_grads_file, 'w') as f:
                        #    f.create_dataset("x", shape=input_gradnorms_shape, 
                        #                     data=input_grad_norms[0].numpy(), compression="lzf", 
                        #                     chunks=(1,) + input_gradnorms_shape[-3:] if len(input_gradnorms_shape) == 4 else (1,1,) + input_gradnorms_shape[-3:])
                        #    if len(input_grad_norms) > 1:
                        #        f.create_dataset("x_mark", shape=input_gradnorms_shape[:-1] + (5,), 
                        #                         data=input_grad_norms[1].numpy(), compression="lzf", 
                        #                     chunks=(1,) + input_gradnorms_shape[-3:] if len(input_gradnorms_shape) == 4 else (1,1,) + input_gradnorms_shape[-3:])
                        #        f.create_dataset("y_mark", shape=input_gradnorms_shape[:-1] + (5,), 
                        #                         data=input_grad_norms[2].numpy(), compression="lzf", 
                        #                     chunks=(1,) + input_gradnorms_shape[-3:] if len(input_gradnorms_shape) == 4 else (1,1,) + input_gradnorms_shape[-3:])

                        save_dict = {"batch": torch.tensor(i), "gradnorms": grad_norms_per_timestep}
                        torch.save(save_dict, gradnorms_file)

                        loss.backward(retain_graph=False)
                        
                        # No optim.step() between batches with detach()
                        for param in self.model.parameters():
                            if not param.grad is None:
                                param.grad.detach()
                        batch_x.grad.detach()
                        #batch_x_mark.grad.detach()
                        #batch_y_mark.grad.detach()

            if self.args.inspect_backward_pass is None and self.args.calculate_acf is None:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
                
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(test_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)
            
        self.model.to(torch.device('cpu'))
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        print ('GPU availability for test:', torch.cuda.is_available())
        try:
            self.model.to(torch.device('cuda'))
        except Exception:
            pass

        return self.model

    def interpolate1d(self, tensor, length):
        
        indices_out = torch.arange(length).to(tensor.device) * tensor.shape[0] / length
        lower_idxs, upper_idxs = torch.floor(indices_out).long(), torch.ceil(indices_out).long().clamp(0, tensor.shape[0]-1)
        
        m = (tensor[upper_idxs] - tensor[lower_idxs]) / (upper_idxs - lower_idxs) #+ 1e-9)
        c = tensor[lower_idxs] - (m * lower_idxs)
        
        selection = tensor[upper_idxs]
        bilinear_interpolation = m * indices_out + c
        out = torch.where(upper_idxs == lower_idxs, selection, bilinear_interpolation)

        return out

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model', setting)
            
            fpath = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            while os.path.islink(fpath):
                fpath = os.readlink(fpath)
            
            if self.args.load_from_chkpt is None:
                if not torch.cuda.is_available():
                    state_dict = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=torch.device('cpu'))
                else:
                    state_dict = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            
                if 'module.' in next(iter(state_dict)):
                    from collections import OrderedDict
                    state_dict_new = OrderedDict()
                    for k, v in state_dict.items():
                        state_dict_new[k[7:]] = v
                    state_dict = state_dict_new
                self.model.load_state_dict(state_dict)
                print ('loaded model from checkpoints directory')
            else:
                print ('loaded model from load_from_chkpt or json file')

        if not self.args.calculate_acf is None:
            autocorrs = []
        
        preds = []
        trues = []
        if test_data.get_num_features() < 7: # RAM-specific
            metric_avg = False
        else:
            #preds = np.zeros((self.args.pred_len, test_data.get_num_features()))
            #trues = np.zeros((self.args.pred_len, test_data.get_num_features()))
            #count = 0
            metric_avg = True

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.args.features == "SM":
            self.args.features = 'M'
            _, test_loader = self._get_data(flag="test")
            self.args.features = "SM"
        
        if seq_len > 720 or pred_len > 720:
             feature_type = setting.split("ft")[-1].split('_')[0]
             if 'M' in feature_type:
                shape = (self.args.pred_len, num_features)
            else:
                shape = (self.args.pred_len, 1)
             print ('.'*50, "\n\n\t\tEvaluating over %d features\n\n" % num_features, '.'*50)
            mse_running_avg = RunningAvgMetrics(shape, "mse")
            mae_running_avg = RunningAvgMetrics(shape, "mae")
            cpu_eff = True
        else:
            cpu_eff = False

        epoch_time = time.time()
       
       if self.args.inspect_backward_pass is None:
            self.model.eval()
        else:
            self.model.train()

        colors = np.array(cmap(np.linspace(0., 1., self.args.pred_len+1)))
        
        if "Weather_Station" in self.args.data:
            weather_metrics = WeatherMetricsCalculator()

        with torch.no_grad() if self.args.inspect_backward_pass is None else torch.enable_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                print ('batch %d/%d' % (i, len(test_loader)), end='\r')
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Weather dataset includes percentiles
                if isinstance(batch_x, tuple):
                    batch_x, percentile = batch_x
                    percentile = percentile.squeeze(0)    
                
                if not self.args.model == "SpaceTime":
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    if not self.args.model == "MultiResolutionDDPM":
                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            if self.args.features == "SM":
                                outputs = torch.cat([self.model(batch_x[...,idx:idx+1], batch_x_mark, dec_inp, batch_y_mark)[0] \
                                                for idx in range(batch_x.shape[-1])], dim=-1)
                            else:
                                if self.args.model == "CycleNet":
                                    outputs = self.model(batch_x, batch_cycle)[0]
                                elif self.args.model == "SpaceTime":
                                    (outputs, _), _ = self.model(batch_x)
                                elif self.args.model == "MultiResolutionDDPM":
                                    self.model.test_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            if self.args.features == "SM":
                                outputs = torch.cat([self.model(batch_x[...,idx:idx+1], batch_x_mark, dec_inp, batch_y_mark) \
                                                for idx in range(batch_x.shape[-1])], dim=-1)
                            else:
                                if self.args.model == "CycleNet":
                                    outputs = self.model(batch_x, batch_cycle)
                                elif self.args.model == "SpaceTime":
                                    (outputs, _), _ = self.model(batch_x)
                                elif self.args.model == "MultiResolutionDDPM":
                                    self.model.test_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                else:
                    if self.args.output_attention:
                        if self.args.features == "SM":
                            outputs = torch.cat([self.model(batch_x[...,idx:idx+1], batch_x_mark, dec_inp, batch_y_mark)[0] \
                                                    for idx in range(batch_x.shape[-1])], dim=-1)
                        else:
                            if self.args.model == "CycleNet":
                                outputs = self.model(batch_x, batch_cycle)[0]
                            elif self.args.model == "SpaceTime":
                                (outputs, _), _ = self.model(batch_x)
                            elif self.args.model == "MultiResolutionDDPM":
                                self.model.test_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        if self.args.features == "SM":
                            if self.args.model == "MultiResolutionDDPM":
                                outputs = torch.cat([self.model.test_forward(batch_x[...,idx:idx+1], batch_x_mark, batch_y[...,idx:idx+1], batch_y_mark) \
                                                        for idx in range(batch_x.shape[-1])], dim=-1)
                            else:
                                outputs = torch.cat([self.model(batch_x[...,idx:idx+1], batch_x_mark, dec_inp, batch_y_mark) \
                                                        for idx in range(batch_x.shape[-1])], dim=-1)
                        else:
                            if self.args.model == "CycleNet":
                                outputs = self.model(batch_x, batch_cycle)
                            elif self.args.model == "SpaceTime":
                                (outputs, _), _ = self.model(batch_x)
                            elif self.args.model == "MultiResolutionDDPM":
                                outputs = self.model.test_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if not self.args.calculate_acf is None:
                    for b in range(batch_y.shape[0]):
                        feature_autocorrs = []
                        for f in range(batch_y.shape[2]):
                            autocorr_pred = acf(outputs[b,:,f].detach().cpu().numpy(), nlags=self.args.calculate_acf)
                            autocorr_gt = acf(batch_y[b,-self.args.pred_len:,f].detach().cpu().numpy(), nlags=self.args.calculate_acf)
                            
                            if not (np.any(np.isnan(autocorr_pred)) or np.any(np.isnan(autocorr_gt))):
                                feature_autocorrs.append([autocorr_pred, autocorr_gt])
                            else:
                                break
                            
                            if f == batch_y.shape[2]-1:
                                autocorrs.append(feature_autocorrs)
                
                elif not self.args.inspect_backward_pass is None:
                    
                    grad_cams = [] # Uses last layer features - per layer activation maps
                    grad_cams_heatmap = np.zeros((self.args.pred_len+1, self.args.seq_len))

                    for h in range(self.args.pred_len+1):
                        criterion = self._select_criterion(backward_pass_inspect_cutoff=h, 
                                        inspect_type=self.args.inspect_backward_pass, horizon=self.args.pred_len)
                        loss = criterion(outputs, batch_y)
                            
                        loss.backward(retain_graph=True)
                        grad_cams = []
                        for n, param in self.model.named_parameters():
                            if not param.grad is None:
                                #indices = list(set(list(range(len(param.grad.shape)))) - set(
                                #    [idx for idx, x in enumerate(param.grad.shape) if x == self.args.d_model]))
                                #if len(indices) == 0 and len(param.grad.shape) > 1:
                                #    indices = [1]
                                
                                if len(param.grad.shape) > 1:
                                    indices = [0] if len(param.grad.shape) > 1 else [] # All Linear layers have weights with input dimension in index 1
                                    if len(indices) > 0:
                                        #gradcam_wt = (param.grad.mean(dim=indices) * param.grad).mean(dim=indices)
                                        # Length alone doesn't fully represent the feature
                                        gradcam_wt = (param.grad.mean() * param).mean(dim=indices)
                                    else:
                                        gradcam_wt = param.grad.mean() * param

                                    gradcam_wt = self.interpolate1d(gradcam_wt, self.args.pred_len)

                                    grad_cams.append(gradcam_wt)
                        
                        grad_cams = F.softmax(torch.stack(grad_cams).mean(dim=0))
                        grad_cams_heatmap[h] = grad_cams.detach().cpu().numpy()

                        for param in self.model.parameters():
                            if not param.grad is None:
                                param.grad.fill_(0)
                    
                    loss.backward(retain_graph=False)
                    # No zero_grad between batches with detach()
                    for param in self.model.parameters():
                        if not param.grad is None:
                            param.grad.detach()

                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                if not metric_avg:
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                
                #plt.imshow(grad_cams_heatmap, cmap="gist_heat")
                #for idx in range(len(grad_cams_heatmap)):
                #    plt.plot(np.arange(self.args.seq_len), grad_cams_heatmap[idx], color=colors[idx])
                #    plt.savefig("sample.png"); 
                #plt.savefig("sample.png")
                #exit()

#                if epoch > 0:
#                    for idx in range(self.args.pred_len+1):
#                        if self.args.inspect_backward_pass == "backward": # 0:idx entries are 0
#                            print ("Grad norm for H: %d->%d: %.5f" % (idx, self.args.pred_len,
#                                                                        grad_norms_per_timestep["backward"][idx].mean()))
#                        else:
#                            print ("Grad norm for H: %d->%d: %.5f" % (1, idx,
#                                                                        grad_norms_per_timestep["forward"][idx].mean()))
#                    exit()
                
                if "Weather_Station" in self.args.data:
                    weather_metrics.update(pred, true, percentile)
                else:
                    if not cpu_eff:
                        preds.append(pred)
                        trues.append(true)
                    else:
                        mse_running_avg.add_to_mean(pred, true)
                        mae_running_avg.add_to_mean(pred, true)
                
                if i % 20 == 0 and not metric_avg:
                    
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, :], true[0, :, :]), axis=0)
                    pd = np.concatenate((input[0, :, :], pred[0, :, :]), axis=0)
                    
                    if "Weather_Station" in self.args.data:
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'), test_data.vnames)
                    else:
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
            
                if i==5 and self.args.gpu_memory_usage:
                    from model_size import model_size
                    print ("MEMORY: Model Size: %s: %fMB" % (self.args.model, model_size(self.model)))
                    print ("MEMORY: GPU summary (in GB) after backward pass:")
                    print ("allocated per data pt:", torch.cuda.memory_allocated(self.device)/(1024.*1024*1024*self.args.batch_size))
                    print ("reserved per data pt:", torch.cuda.memory_reserved(self.device)/(1024.*1024*1024*self.args.batch_size))
                    print ("Time per epoch: %f seconds" % ((time.time()-epoch_time)*len(test_loader)/(6.)))
                    exit()               
                        
            if not self.args.calculate_acf is None:
                print ("Autocorrelation for %s pred:" % self.args.model, np.array(autocorrs)[:,:,0:1,:].mean(axis=(0,1,2)))
                print ("Autocorrelation for %s gt:" % self.args.model, np.array(autocorrs)[:,:,1:2,:].mean(axis=(0,1,2)))
                exit()
 
        if not cpu_eff:
            if not metric_avg and not "Weather_Station" in self.args.data:
                preds = np.concatenate(preds, axis=0)
                trues = np.concatenate(trues, axis=0)
                print('test shape:', preds.shape, trues.shape)
                mae, mse, rmse, mape, mspe = metric(preds, trues)
                print('mse:{}, mae:{}'.format(mse, mae))
                print ('result:', self.args.target, ((preds-trues)**2).mean(axis=(0,1)), np.abs(preds-trues).mean(axis=(0,1)))
        
            else:
                mse = [((x-y)**2).mean(dim=-1) for x,y in zip(preds, trues)]
                mae = [torch.abs(x-y).mean(dim=-1) for x,y in zip(preds, trues)]
                #mse = [((x-y)**2).mean() for x,y in zip(preds, trues)]
                #mae = [torch.abs(x-y).mean() for x,y in zip(preds, trues)]
                mse = torch.cat(mse)
                mae = torch.cat(mae)
                #mse = torch.tensor(mse).mean()
                #mae = torch.tensor(mae).mean()
                print (mse.shape, mae.shape)
                print('mse:{}, mae:{}'.format(mse.mean(), mae.mean()))
                #preds = torch.cat(preds)
                #trues = torch.cat(trues)

        else:
            mse = mse_running_avg.get_mean()
            mae = mae_running_avg.get_mean()
            
            print ("mse:{}, mae:{}".format(mse.mean(), mae.mean()))

        # result save
        #folder_path = './results/' + setting + '/'
        #if not os.path.exists(folder_path):
        #    os.makedirs(folder_path)
        
        #f = open("result.txt", 'a')
        #f.write(setting + "  \n")
        #f.write('mse:{}, mae:{}'.format(mse, mae))
        #f.write('\n')
        #f.write('\n')
        #f.close()

        #np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        #np.save(folder_path + 'pred.npy', preds)
        #np.save(folder_path + 'true.npy', trues)
        
        plt.rcParams["figure.figsize"] = 5,2
        
        if not os.path.isdir("error_heatmap_std"):
            os.mkdir("error_heatmap_std")
            heatmap_idx = 0
        else:
            heatmaps = glob.glob(os.path.join("error_heatmap_std", "*%s_%d*" % (self.args.model, self.args.pred_len)))
            if len(heatmaps) == 0:
                heatmap_idx = 0
            else:
                heatmaps = sorted([int(x.split('_')[-1].split('.')[0]) for x in heatmaps])
                heatmap_idx = heatmaps[-1] + 1

        for start in [0]:
            x = np.linspace(start, self.args.pred_len, num=self.args.pred_len-start)
            
            if not cpu_eff:
                if not metric_avg:
                    y = np.mean((preds-trues)**2, axis=(0,2))[start:]
                else:
                    y = mse.mean(dim=0).cpu().numpy()[start:]
            else:
                y = mse_running_avg.get_mean().mean(axis=-1)[start:]

            fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

            extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
            ax.imshow(y[np.newaxis,:], cmap="inferno", aspect="auto", extent=extent)
            ax.set_yticks([])
            ax.set_xlim(extent[0], extent[1])

            np.save(os.path.join("error_heatmap_std", "%s_%s_%d_%s" % (self.args.data, self.args.model, self.args.pred_len, heatmap_idx)), y)
            ax2.plot(x,y)

            plt.tight_layout()
            plt.savefig("%s_%s_heatmap_%d_M.pdf" % (self.args.data, self.args.model, self.args.pred_len), dpi=300, bbox_inches="tight")

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

# Train diffusion models
def log_normal(x, mu, var):
    """Logarithm of normal distribution with mean=mu and variance=var
       log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

    Args:
       x: (array) corresponding array containing the input
       mu: (array) corresponding array containing the mean
       var: (array) corresponding array containing the variance

    Returns:
       output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
    """
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    # return -0.5 * torch.sum(
    #     np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)
    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)
