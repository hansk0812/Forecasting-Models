import torch
from torch.nn import functional as F
import einops

from .SSMBase import SSMBase

def pascals_triangle_evens_negated(n):

    # Compute binomial coeffs for all rows up to n
    line = torch.zeros(n, n).float()
    line[:, 0] = 1.
    for j in range(1, n):      # For all rows,
        for k in range(0, j):  # Compute C(j, k)
            # Coefficients are binomial coeffs, 
            # C(n, k + 1) = C(n, k) * (n - k) / (k + 1)
            negate = 2 * k % 2 - 1  # Negate even elements
            line[j][k+1] += (line[j][k] * (j - k) / (k + 1)) * negate
    return line

class PreprocessSSM(SSMBase):
    """
    Computes both order-N differencing and moving average residuals over input sequence
    """
    def __init__(self, 
                 max_diff_order=4, 
                 min_avg_window=4, 
                 max_avg_window=720,
                 num_kernels=8,
                 kernel_repeat=16,
                 **kwargs):
        self.max_diff_order = max_diff_order
        self.min_avg_window = min_avg_window
        self.max_avg_window = max_avg_window
        self.n_ma_kernels = (num_kernels - self.max_diff_order) * kernel_repeat
        kwargs['num_heads'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = False
        kwargs['skip_connection'] = False
        # Set kwargs['kernel_repeat'] to number of model num_kernels
        super().__init__(num_kernels=num_kernels, kernel_repeat=kernel_repeat, **kwargs)
        
    def init_weights(self):
        diff_kernel = self.init_differencing_weights()
        diff_kernel = einops.repeat(diff_kernel, "i j -> (i1 i) j", i1=self.kernel_repeat)
        ma_r_kernel = self.init_moving_average_weights()  # Shape: (kr x nk) x hd
        self.register('diff_kernel', diff_kernel, trainable=False)
        self.register('ma_r_kernel', ma_r_kernel, trainable=False)
        
    def init_differencing_weights(self):
        kernel = torch.zeros(self.max_diff_order, self.max_diff_order).float()
        diff_coeffs = pascals_triangle_evens_negated(self.max_diff_order).float()
        kernel[:, :self.max_diff_order] += diff_coeffs
        return kernel
    
    def init_moving_average_weights(self):
        ma_window = torch.randint(low=self.min_avg_window,
                                  high=self.max_avg_window,
                                  size=(1, self.n_ma_kernels))
        # Compute moving average kernel 
        max_window = self.max_avg_window
        kernel = torch.zeros(self.n_ma_kernels, max_window)
        kernel[:, 0] = 1.
        
        moving_avg = 1. / ma_window
        for ix, window in enumerate(ma_window[0]):
            kernel[ix, :window] -= moving_avg[:1, ix]
        return kernel

    def get_kernel(self, x):
        """
        x: B x D x L
        """
        b, d, l = x.shape
        l = max(l, self.diff_kernel.shape[1])
        
        # Pad kernels to input length
        diff_kernel = F.pad(self.diff_kernel, (0, l - self.diff_kernel.shape[1]), 'constant', 0)
        ma_r_kernel = F.pad(self.ma_r_kernel, (0, l - self.ma_r_kernel.shape[1]), 'constant', 0)

        # Combine kernels
        diff_kernel = diff_kernel.reshape((self.kernel_repeat, -1, diff_kernel.shape[-1]))
        ma_r_kernel = ma_r_kernel.reshape((self.kernel_repeat, -1, ma_r_kernel.shape[-1]))
        
        kernel = torch.cat([diff_kernel, ma_r_kernel], dim=1)
        kernel = einops.repeat(kernel, "b d l -> (b h d) l", h=self.head_dim)
        return kernel
    
    def forward(self, x):
        x = x.transpose(1, 2)
        k = self.get_kernel(x)
        y = self.fft_conv(x, k)
        return y.transpose(1, 2)

