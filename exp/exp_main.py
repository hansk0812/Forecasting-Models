import os
import time
import warnings
import numpy as np

import traceback

import torch
import torch.nn as nn
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

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)

from scipy.signal import correlate
from statsmodels.tsa.stattools import acf

from tqdm import tqdm

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):

        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Triformer': Triformer,
            'FiLM': FiLM,
            'DLinear': DLinear,
            'NLinear': NLinear,
            #'NLinearLHF': NLinearLHF,
            'NHITS': NHITS,
            'TiDE': TiDE,
            'NBEATS': NBEATS,
            'Pyraformer': Pyraformer,
            'SAMformer': SAMformer,
            'CycleNet': CycleNet,
            'SpaceTime': SpaceTime,
            'MultiResolutionDDPM': MultiResolutionDDPM
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
            try:
                model.load_state_dict(torch.load(self.args.load_from_chkpt, weights_only=True))
            except Exception:
                pass
            print ("\n", "."*50, "\n\nLoaded initial model from %s\n\n" % self.args.load_from_chkpt, "."*50)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, backward_pass_inspect_cutoff=None, inspect_type=None, horizon=None):
        if backward_pass_inspect_cutoff is None:
            criterion = nn.MSELoss()
        else:
            assert not horizon is None, "Interpreters!"
            criterion = BackwardPassInspectLoss(horizon, backward_pass_inspect_cutoff, 
                                                inspect_type, device=self.device, loss=self.args.loss)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
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
                    loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = torch.mean(torch.stack(total_loss))
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        import time
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        batch_start = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()
            train_loader = tqdm(train_loader, leave=False)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
               
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
                    train_loss.append(loss.item())
                
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
                    loss.backward()
                    model_optim.step()
            
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
        self.model.to(torch.device('cuda'))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model', setting)
            
            fpath = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            while os.path.islink(fpath):
                fpath = os.readlink(fpath)

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
            print ('loaded model')

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.args.features == "SM":
            self.args.features = 'M'
            _, test_loader = self._get_data(flag="test")
            self.args.features = "SM"
        
        epoch_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                print ('batch %d/%d' % (i, len(test_loader)), end='\r')
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
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
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                elif self.args.model == "SpaceTime":
                                    (outputs, _), _ = self.model(batch_x)
                                elif self.args.model == "MultiResolutionDDPM":
                                    self.model.test_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)
                                else:
                                    outputs = self.model(batch_x, batch_cycle)[0]
                        else:
                            if self.args.features == "SM":
                                outputs = torch.cat([self.model(batch_x[...,idx:idx+1], batch_x_mark, dec_inp, batch_y_mark) \
                                                for idx in range(batch_x.shape[-1])], dim=-1)
                            else:
                                if self.args.model == "CycleNet":
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                elif self.args.model == "SpaceTime":
                                    (outputs, _), _ = self.model(batch_x)
                                elif self.args.model == "MultiResolutionDDPM":
                                    self.model.test_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)
                                else:
                                    outputs = self.model(batch_x, batch_cycle)
                else:
                    if self.args.output_attention:
                        if self.args.features == "SM":
                            outputs = torch.cat([self.model(batch_x[...,idx:idx+1], batch_x_mark, dec_inp, batch_y_mark)[0] \
                                                    for idx in range(batch_x.shape[-1])], dim=-1)
                        else:
                            if self.args.model == "CycleNet":
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            elif self.args.model == "SpaceTime":
                                (outputs, _), _ = self.model(batch_x)
                            elif self.args.model == "MultiResolutionDDPM":
                                self.model.test_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)
                            else:
                                outputs = self.model(batch_x, batch_cycle)[0]

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
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
            
                if i==5 and self.args.gpu_memory_usage:
                    from model_size import model_size
                    print ("MEMORY: Model Size: %s: %fMB" % (self.args.model, model_size(self.model)))
                    print ("MEMORY: GPU summary (in GB) after backward pass:")
                    print ("allocated per data pt:", torch.cuda.memory_allocated(self.device)/(1024.*1024*1024*self.args.batch_size))
                    print ("reserved per data pt:", torch.cuda.memory_reserved(self.device)/(1024.*1024*1024*self.args.batch_size))
                    print ("Time per epoch: %f seconds" % ((time.time()-epoch_time)*len(test_loader)/(6.)))
                    exit()
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        print ('result:', self.args.target, ((preds-trues)**2).mean(axis=(0,1)), np.abs(preds-trues).mean(axis=(0,1)))
    
        plt.rcParams["figure.figsize"] = 5,2
        
        start=0
        x = np.linspace(start,720,num=720-start)
        y = np.mean((preds-trues)**2, axis=(0,2))[start:]
        
        fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

        extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
        ax.imshow(y[np.newaxis,:], cmap="inferno", aspect="auto", extent=extent)
        ax.set_yticks([])
        ax.set_xlim(extent[0], extent[1])

        ax2.plot(x,y)

        plt.tight_layout()
        plt.savefig("%s_heatmap_720_M_start%d.png" % (self.args.model, s))

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
