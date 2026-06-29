import os
import torch
import numpy as np

from torch import nn

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
            
        if args.use_multi_gpu and len(args.devices) > 1:
            
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.device("cuda:%s" % os.environ.get("LOCAL_RANK"))
            self.device = local_rank
            torch.cuda.set_device(local_rank)

			# Use torchrun --nproc_per_node NUM_GPUS <script.py ...>
            self.model = nn.parallel.DistributedDataParallel(self.model.to(local_rank), device_ids=[local_rank])
            #torch.distributed.init_process_group(backend="nccl", 
            #                                     rank=0, 
            #                                     world_size=len(args.devices), 
            #                                     init_method="tcp://localhost:12355")
            #self.model = nn.parallel.DistributedDataParallel(self.model)
 
    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, *args, **kwargs):
        pass

    def vali(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass
