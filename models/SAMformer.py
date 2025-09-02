import torch
from torch import nn

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) https://openreview.net/pdf?id=cGDAkQo1C0p
    https://github.com/ts-kim/RevIN
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.nn.init.constant_(torch.ones(self.num_features), 0.01))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.seq_len != config.pred_len:
            self.embedding = nn.Linear(config.seq_len, config.pred_len)
        
        self.embed_flag = config.seq_len != config.pred_len
        self.revin = RevIN(num_features=config.pred_len)
        self.compute_keys = nn.Linear(config.pred_len, config.d_model)
        self.compute_queries = nn.Linear(config.pred_len, config.d_model)
        self.compute_values = nn.Linear(config.pred_len, config.pred_len)
        self.linear_forecaster = nn.Linear(config.pred_len, config.pred_len)
        self.use_revin = True

    def forward(self, x, v1, v2, v3, v4=None, v5=None, v6=None):
        
        if self.embed_flag:
            x = self.embedding(x.transpose(1,2))
        else:
            x = x.transpose(1,2)
       
        # RevIN Normalization
        if self.use_revin:
            x_norm = self.revin(x, mode='norm') # (n, D, L)
        else:
            x_norm = x
        # Channel-Wise Attention
        queries = self.compute_queries(x_norm) # (n, D, hid_dim)
        keys = self.compute_keys(x_norm) # (n, D, hid_dim)
        values = self.compute_values(x_norm) # (n, D, L)
        if hasattr(nn.functional, 'scaled_dot_product_attention'):
            att_score = nn.functional.scaled_dot_product_attention(queries, keys, values) # (n, D, L)
        else:
            att_score = scaled_dot_product_attention(queries, keys, values) # (n, D, L)
        out = x_norm + att_score # (n, D, L)
        # Linear Forecasting
        out = self.linear_forecaster(out) # (n, D, H)
        # RevIN Denormalization
        if self.use_revin:
            out = self.revin(out, mode='denorm').transpose(1, 2) # (n, D, H)
        return out
