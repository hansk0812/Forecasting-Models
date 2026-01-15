#TODO: Run smallest batch size over many trained models of different batch sizes for scale parameter trend

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import optim

import argparse
import os
import glob

import numpy as np

from collections import OrderedDict

class PolynomialCurve(nn.Module):

    def __init__(self, num_batchsizes=1):
        
        super().__init__()

        self.W = nn.init.kaiming_normal_(nn.Parameter(torch.zeros(num_batchsizes, 1)))
        self.b = nn.init.kaiming_normal_(nn.Parameter(torch.zeros(num_batchsizes, 1)))
        self.k = nn.init.kaiming_normal_(nn.Parameter(torch.zeros(num_batchsizes, 1)))
        
        self.scale = 1
        #self.scale = nn.init.normal_(nn.Parameter(torch.zeros(num_batchsizes, 1)))

    def forward(self, x):
        return (self.W/torch.pow(x, self.k) + self.b) * self.scale

class SubseriesScale(nn.Module):

    def __init__(self, num_batchsizes):
        
        super().__init__()

        self.m = nn.init.kaiming_normal_(nn.Parameter(torch.zeros(num_batchsizes, 1)))
        self.c = nn.init.kaiming_normal_(nn.Parameter(torch.zeros(num_batchsizes, 1)))

    def forward(self, x):
        return self.m * x + self.c

class CurveData(Dataset):

    def __init__(self, folder, model, batch_sizes, epoch, h, interpolated=True):
        
        data_batchsizes = OrderedDict()
        for b in batch_sizes:
            
            if model in ["CycleNet", "NLinear"]:
                fnames = glob.glob(os.path.join(folder, model + "_*_%d_batchsizes.txt" % h))
            else:
                if epoch == 0:
                    fnames = glob.glob(os.path.join(folder, model + "Random_*_%d_batchsizes.txt" % h))
                else:
                    fnames = glob.glob(os.path.join(folder, model + "Batch%dEpoch%s_*_%d_batchsizes.txt" % (b, epoch, h)))
            fnames = sorted(fnames, key=lambda x: int(x.split('_')[1]))

            subseries = OrderedDict()
            subseries_keys = []
            for f in fnames:
                key = int(f.split('_')[1])
                grads = OrderedDict()
                all_batches = []
                with open(f, 'r') as f:
                    for x in f.readlines():
                        if "Grad" in x:
                            batch = int(x.split(':')[0])
                            
                            # Model norm for now
                            g = float(x.split(": ")[-1].strip())
                            if '=' in g:
                                g_keys = [x.split('=')[0].strip() for x in g.split(' ')]
                                g_vals = [float(x.split('=')[1].strip()) for x in g.split(' ')]
                                g = np.array(g_vals).mean()

                            grads[batch] = g
                            all_batches.append(batch)
                subseries_keys.append(key)
                subseries[key] = grads
            data_batchsizes[b] = subseries
        
        # |batchsizes in model| x |h in H| x |gradnorm batchsizes|
        data = torch.zeros((len(batch_sizes), len(subseries_keys), len(all_batches)))

        # normalize by maximum gradnorm at min(gradnorm batchsizes)
        for idx, b in enumerate(batch_sizes):
            for jdx, n_t in enumerate(subseries_keys):
                for kdx, n_g in enumerate(all_batches):
                    data[idx, jdx, kdx] = data_batchsizes[b][n_t][n_g]
                    
                    if interpolated:
                        data[idx, jdx, kdx] /= data_batchsizes[b][n_t][all_batches[0]]
        
        if interpolated:
            if model == "NLinear": # remove first two subseries which show stochasticity
                data[:,0,:] = data[:,2,:]
                data[:,1,:] = data[:,2,:]

        self.data = data
        self.all_batches = torch.tensor(all_batches) / float(all_batches[-1])
        self.subseries_keys = torch.tensor([int(x) for x in subseries_keys]).float()
        if not interpolated:
            self.subseries_keys /= self.subseries_keys.max()
        
        self.interpolated = interpolated

        self.x_mean, self.x_std = self.all_batches.mean(), self.all_batches.std()
        self.y_mean, self.y_std = data.mean(dim=(0,2)).unsqueeze(0).unsqueeze(-1), data.std(dim=(0,2)).unsqueeze(0).unsqueeze(-1)
        
        # Function range doesn't allow normalization around 0
        #self.data = (self.data - self.y_mean) / (self.y_std + 1e-7)
        #self.all_batches = (self.all_batches - self.x_mean) / (self.x_std + 1e-7)
        #from pprint import pprint
        #pprint (self.data)
        #pprint (self.all_batches)

#        from matplotlib import pyplot as plt
#        for batch_idx in range(len(batch_sizes)):
#        ##batch_idx = 2
#            for idx in range(data.shape[1]):
#                plt.plot(list(range(len(all_batches))), data[batch_idx,idx,:].numpy())# / data[batch_idx,idx,0])
#        plt.show()
#        exit()

    def __getitem__(self, idx):

        if self.interpolated:
            # add noise
            data_noised = self.data[:,idx,:] + torch.randn(self.data[:,idx,:].shape) * self.data[:,idx,:].min() * 0.3
            return torch.stack([self.all_batches for _ in range(self.data.shape[0])]), data_noised #self.data[:,idx,:]
        else:
            return torch.stack([self.subseries_keys[idx] for _ in range(self.data.shape[0])]), self.data[:,idx,0]

    def __len__(self):
        return self.data.shape[1]

class ScaleFactors(Dataset):

    def __init__(self, folder, model, batch_size, epoch, h):

        self.min_batch_size = batch_size

        fnames = glob.glob(os.path.join(folder, model + "ScaleBatch%d_*_%d_batchsizes.txt" % (batch_size, h)))

        subseries = OrderedDict()
        for f in fnames:

            key = int(f.split('_')[1])

            grads = OrderedDict()
            with open(f, 'r') as f:
                for batch_line in f.readlines():
                    batch = int(batch_line.split(':')[0])
                    
                    g = batch_line.split(": ")[1]
                    if '=' in g:
                        keys = [x.split('=')[0].strip() for x in g.split(' ')]
                        values = [float(x.split('=')[1].strip()) for x in g.split(' ')]
                        g = np.array(values).mean()
                    
                    grads[batch] = g
            subseries[key] = grads

        self.subseries_keys = list(subseries.keys())
        self.batch_sizes = list(subseries[self.subseries_keys[0]].keys())

        # |h in H| x |gradnorm batchsizes|
        self.data = torch.zeros((len(self.subseries_keys), len(self.batch_sizes)))
        for idx in range(self.data.shape[0]):
            for jdx in range(self.data.shape[1]):
                self.data[idx][jdx] = subseries[self.subseries_keys[idx]][self.batch_sizes[jdx]]

        self.subseries_keys = np.array(self.subseries_keys, dtype=np.float32)
        self.subseries_keys /= self.subseries_keys.max()
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.stack([self.subseries_keys for _ in range(self.data.shape[1])]), self.data[idx,:]

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("folder", help="Path of gradnorm average txt files")
    ap.add_argument("model", help="Single model's name")
    ap.add_argument("h", help="Horizon size", type=int)
    ap.add_argument("epoch", help="Epoch to fit", type=int)
    ap.add_argument("--batch_sizes", nargs='+', help="Batch sizes available for model", type=int, required=True)
    ap.add_argument("--alpha", type=float, help="MSE weight in interpolation with MAE")
    ap.add_argument("--scale_parameter", help="(W,k,b) format to train for scale parameter", default=None)
    ap.add_argument("--scale_batchsizes_test", help="Use flag to run scale line fit over all batchsizes at min(batch)=100", action="store_true")
    ap.add_argument("--test", help="Test all batchsizes using (W,k,b) for polynomial curve and (m,c) for linear scale", default=None)

    args = ap.parse_args()
    
    #args.epoch = "Random" if args.epoch == 0 else str(args.epoch)

    if args.scale_batchsizes_test:

        dataset = ScaleFactors(args.folder, args.model, 100, args.epoch, args.h)
        models = [SubseriesScale(1) for _ in range(dataset.data.shape[1])]

        dataset = DataLoader(dataset, batch_size=dataset.data.shape[1], shuffle=True)
        
        if not args.test is None:
            m, c = [float(x) for x in args.test[1:-1].split(',')]
            for model in models:
                model.m, model.c = nn.Parameter(torch.tensor([[m]])), \
                                   nn.Parameter(torch.tensor([[c]]))
            epochs = 0
        else:
            epochs = 20000
            optimizers = [optim.Adam() for _ in range(dataset.data.shape[1])]
 
        for e in range(epochs):
            for x, y in dataset:

                [optimizer.zero_grad() for optimizer in optimizers]

                y_pred = torch.stack([models[idx](x[:,idx]) for idx in range(dataset.data.shape[1])])
                y_pred = y_pred.transpose(0, 1)

                mse_loss = args.alpha * F.mse_loss(y, y_pred)
                mae_loss = args.alpha * F.l1_loss(y, y_pred)

                loss = mse_loss + mae_loss
                loss.backward()
                [optimizer.step() for optimizer in optimizers]
        
        for idx in range(dataset.data.shape[1]):
            print ("Batch size %d: m * x + c " % dataset.batch_sizes[idx], "m=%.5f" % models[idx].m, "c=%.5f" % models[idx].c)
        
        # 
        y_preds = []
        for batch_idx in range(len(dataset.batch_sizes)):
            y_pred = torch.stack([models[batch_idx](dataset.subseries_keys[jdx])[0][0] for jdx in range(dataset.data.shape[0])])
            y_preds.append(y_pred)
        y_preds = torch.stack(y_preds).transpose(0, 1)
        
        from matplotlib import pyplot as plt
        with torch.no_grad():
            for idx in range(dataset.data.shape[1]):
                y_gt = dataset.data[:, idx]
                y_p = y_preds[:, idx]
                x_vals = list(range(50, 50*len(dataset.subseries_keys)+1, 50))
                x_vals[-1] = args.h
                plt.scatter(x_vals, y_gt)
                plt.scatter(x_vals, y_p.numpy())
                plt.show()
       
       exit()

    dataset = CurveData(args.folder, args.model, args.batch_sizes, args.epoch, args.h, interpolated = (args.scale_parameter is None))
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    
    if args.scale_parameter is None:
        
        models = [PolynomialCurve(1) for _ in range(len(args.batch_sizes))]

        if not args.test is None:
            W, k, b = [float(x) for x in args.test[1:-1].split(',')]
            for model in models:
                model.W, model.k, model.b = nn.Parameter(torch.tensor([[W]])), \
                                            nn.Parameter(torch.tensor([[k]])), \
                                            nn.Parameter(torch.tensor([[b]]))
            epochs = 0
        else:
            device = torch.device("cpu")
            models = [model.to(device) for model in models]
            
            optimizers = [optim.Adam(models[idx].parameters()) for idx in range(len(models))]
            alpha = args.alpha
            epochs = 50000
        
        for e in range(epochs):
            for idx, (x, y) in enumerate(dataloader):
                [optimizer.zero_grad() for optimizer in optimizers]
                x, y = x.to(device), y.to(device)
                y_pred = torch.stack([models[idx](x[idx])[0] for idx in range(len(models))])
                mse_loss = alpha * F.mse_loss(y, y_pred)
                mae_loss = (1-alpha) * F.l1_loss(y, y_pred)
                
                print ("Epoch %d Point %d/%d: MSE: %.5f ; MAE: %.5f" % (e, idx, len(dataset), mse_loss, mae_loss), end='\r')
                
                loss = mse_loss + mae_loss 
                loss.backward()
                [optimizer.step() for optimizer in optimizers]
        
        for idx in range(len(args.batch_sizes)):
            print ("Batch size %d: W * x^(-k) + b " % args.batch_sizes[idx], "W=%.5f" % models[idx].W, "k=%.5f" % models[idx].k, "b=%.5f" % models[idx].b)

        y_preds = []
        for jdx in range(dataset.data.shape[1]):
            yp = torch.stack([models[idx](dataset.all_batches)[0] for idx in range(dataset.data.shape[0])])
            y_preds.append(yp)
        y_preds = torch.stack(y_preds).transpose(0, 1)
        print (y_preds.shape, dataset.data.shape)
        
        from matplotlib import pyplot as plt
        with torch.no_grad():
            for idx in range(len(args.batch_sizes)):
                for jdx in range(dataset.data.shape[1]):
                    y_gt = dataset.data[idx,jdx]
                    y_p = y_preds[idx,jdx]
                    plt.plot(list(range(100, 100*len(y_gt)+1, 100)), y_gt) #* dataset.y_std[jdx] + dataset.y_mean[jdx])
                    plt.plot(list(range(100, 100*len(y_p)+1, 100)), y_p.numpy()) #* dataset.y_std[jdx] + dataset.y_mean[jdx])
                    plt.show()
    else:
        
        W, k, b = [float(x) for x in args.scale_parameter[1:-1].split(',')]
        models = [SubseriesScale(1) for _ in range(len(args.batch_sizes))]

        if not args.test is None:
            m, c = [float(x) for x in args.test[1:-1].split(',')]
            for model in models:
                model.m, model.c = nn.Parameter(torch.tensor([[m]])), \
                                   nn.Parameter(torch.tensor([[c]]))
            epochs = 0
        else:
            device = torch.device("cpu")
            models = [model.to(device) for model in models]
            
            optimizers = [optim.Adam(models[idx].parameters()) for idx in range(len(models))]
            alpha = args.alpha
            epochs = 5000
        
        for e in range(epochs):
            for idx, (x, y) in enumerate(dataloader):
                [optimizer.zero_grad() for optimizer in optimizers]
                x, y = x.to(device), y.to(device)
                y_pred = torch.stack([models[idx](x[idx])[0][0] for idx in range(len(models))])
                mse_loss = alpha * F.mse_loss(y, y_pred)
                mae_loss = (1-alpha) * F.l1_loss(y, y_pred)
                
                print ("Epoch %d Point %d/%d: MSE: %.5f ; MAE: %.5f" % (e, idx, len(dataset), mse_loss, mae_loss), end='\r')
                
                loss = mse_loss + mae_loss 
                loss.backward()
                [optimizer.step() for optimizer in optimizers]
        
        for idx in range(len(args.batch_sizes)):
            print ("Batch size %d: m * x + c " % args.batch_sizes[idx], "m=%.5f" % models[idx].m, "c=%.5f" % models[idx].c)

        y_preds = []
        for jdx in range(dataset.data.shape[1]):
            yp = torch.stack([models[idx](dataset.subseries_keys[jdx])[0][0] for idx in range(dataset.data.shape[0])])
            y_preds.append(yp)
        y_preds = torch.stack(y_preds).transpose(0, 1)
        print (y_preds.shape, dataset.data[:,:,0].shape)
        
        from matplotlib import pyplot as plt
        with torch.no_grad():
            for idx in range(len(args.batch_sizes)):
                y_gt = dataset.data[idx,:,0]
                y_p = y_preds[idx]
                x_vals = list(range(50, 50*len(dataset.subseries_keys)+1, 50))
                x_vals[-1] = args.h
                plt.scatter(x_vals, y_gt)
                plt.scatter(x_vals, y_p.numpy())
                plt.show()

