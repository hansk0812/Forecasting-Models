import os
import glob

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

import matplotlib.pyplot as plt
from matplotlib import colormaps
cmap = colormaps["Reds"]

import numpy as np; np.random.seed(1)

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
            
            torch.save(model.state_dict(), "spacetime.pth")
        
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
                    pass

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
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

            gradnorms_dir = "gradnorms_per_batch"
            if os.path.exists("%s/%s_%d_%s.pth" % (gradnorms_dir, self.args.model, self.args.pred_len, self.args.inspect_backward_pass)):
                try:
                    load_dict = torch.load("%s/%s_%d_%s.pth" % (gradnorms_dir, self.args.model, self.args.pred_len, self.args.inspect_backward_pass))
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
                            if not self.args.backward_pass_multivariate:
                                for idx in range(0, self.args.pred_len+1, step):

                                    if skip_zeroes:
                                        if (self.args.inspect_backward_pass == "forward" and idx!=batchsize_timestep):
                                            continue
                                
                                    if self.args.inspect_backward_pass == "backward": # 0:idx entries are 0
                                        norms_str = ' '.join(["%s=%.16E" % (n, Decimal(g.item())) for n, g in \
                                                                zip(layer_names, grad_norms_per_timestep["backward"][idx].mean(axis=0))])
                                        print ("Grad norm for H: %d->%d: %s" % (idx, self.args.pred_len, norms_str))
                                    else:
                                        norms_str = ' '.join(["%s=%.16E" % (n, Decimal(g.item())) for n, g in \
                                                                zip(layer_names, grad_norms_per_timestep["forward"][idx].mean(axis=0))])
                                        print ("Grad norm for H: %d->%d: %s" % (0, idx, norms_str))
                            else:
                                if not os.path.isdir("logs/gradnorms_M"):
                                    os.makedirs("logs/gradnorms_M")

                                # Added multivariate softmax HAM plots
                                for v_idx in range(self.args.enc_in):
                                    
                                    with open("logs/gradnorms_M/%s_%d_forward_gradnorms.txt", 'w') as f:
                                        with open("logs/gradnorms_M/%s_%d_backward_gradnorms.txt", 'w') as g:
                                            
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
                                 
                                for param in self.model.parameters():
                                    if not param.grad is None:
                                        param.grad.fill_(0)
                            
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
                                    
                                    for param in self.model.parameters():
                                        if not param.grad is None:
                                            param.grad.fill_(0)
                                
                                loss = v_loss

                        save_dict = {"batch": torch.tensor(i), "gradnorms": grad_norms_per_timestep}
                        if not os.path.isdir(gradnorms_dir):
                            os.mkdir(gradnorms_dir)
                        torch.save(save_dict, "%s/%s_%d_%s.pth" % (gradnorms_dir, self.args.model, self.args.pred_len, self.args.inspect_backward_pass))

                        loss.backward(retain_graph=False)
                        
                        # No optim.step() between batches with detach()
                        for param in self.model.parameters():
                            if not param.grad is None:
                                param.grad.detach()

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
                    preds.append(pred)
                    trues.append(true)
                
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
 
        if not metric_avg and not "Weather_Station" in self.args.data:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('test shape:', preds.shape, trues.shape)

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

        # result save
        #folder_path = './results/' + setting + '/'
        #if not os.path.exists(folder_path):
        #    os.makedirs(folder_path)
        
        if not metric_avg:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))
            print ('result:', self.args.target, ((preds-trues)**2).mean(axis=(0,1)), np.abs(preds-trues).mean(axis=(0,1)))
        
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
            heatmaps = glob.glob(os.path.join("error_heatmap_std", "*%s*" % self.args.model))
            if len(heatmaps) == 0:
                heatmap_idx = 0
            else:
                heatmaps = sorted([int(x.split('_')[-1].split('.')[0]) for x in heatmaps])
                heatmap_idx = heatmaps[-1] + 1

        for start in [0]:
            x = np.linspace(start, self.args.pred_len, num=self.args.pred_len-start)
            
            if not metric_avg:
                y = np.mean((preds-trues)**2, axis=(0,2))[start:]
            else:
                y = mse.mean(dim=0).cpu().numpy()

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
