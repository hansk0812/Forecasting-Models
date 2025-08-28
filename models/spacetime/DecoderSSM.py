import torch
import torch.nn.functional as F
import opt_einsum as oe
from einops import repeat, rearrange

from .SSMBase import krylov_logarithmic_products
from .EncoderSSM import EncoderSSM


class DecoderSSM(EncoderSSM):
    """
    Closed-loop implementation of Companion SSM:
    - Instantiate A, B, C; so we can compute both:
    - Open-loop inference:   
      -> y_{n + h} = \sum_{i = 0}^{n + h - 1} CA^{n + h - 1 - i} B u_i
    - Closed-loop inference: 
      -> y_{n + h} = C(A + BK)^{h} x_n
                   = C(A + BK)^{h} \sum_{j = 0}^{n - 1} A^{n - 1 - j} B u_j
                   = C(A + BK)^{n + h - 1} x_1
                   = C(A + BK)^{n + h - 1} B u_0
                   = \sum_{i = 0}^{n + h - 1} C(A + BK)^{n + h - 1 - i} B u_i, u_j = 0 for j > 0
    """
    def __init__(self, 
                 lag: int=1,
                 horizon: int=1,
                 use_initial: bool=False,
                 **kwargs):
        self.lag     = lag
        self.horizon = horizon
        self.use_initial = use_initial  # When False, assumes initial hidden_state x_0 = 0. True not implemented
        self.closed_loop = True         # Toggle closed or open-loop forward pass, see self.forward
        self.inference_only = False     # Toggle different behavior during training and test
        kwargs['kernel_repeat'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = True
        kwargs['skip_connection'] = False

        super().__init__(**kwargs)

    def init_kernel_weights(self, kernel_init):
        if kernel_init == 'normal':
            kernel = torch.randn(self.num_kernels, self.kernel_dim)
        else:
            raise NotImplementedError
        return kernel
        
    def init_weights(self):
        super().init_weights()  # Initializes skip connection, A, B, C
        # K matrix
        k = self.init_kernel_weights(self.kernel_init)
        self.register("k", k, trainable=True)
    
    def get_companion_matrix(self, p):
        # Construct companion matrix
        return self.shift_matrix + \
            oe.contract('h i, h j -> h j i', self.p_padding, p)
    
    def fft_conv_d(self, u, v):
        L   = u.shape[-1]
        u_f = torch.fft.rfft(u, n=2*L, dim=2) # (B H L)
        v_f = torch.fft.rfft(v, n=2*L, dim=2) # (H D L)

        y_f = oe.contract('b h l, h d l -> b h l d', u_f, v_f) 
        y   = torch.fft.irfft(y_f, n=2*L, dim=2)[:, :, :L, :] # (B H L D)
        return y
    
    def forward(self, u):
        """
        During training, call this function twice to compute closed-loop and open-loop
        -> minimize the closed-loop?
        """
        u = u.transpose(1, 2)
        b, d, l = u.shape
        l_horizon = self.horizon
        
        # Normalize just the non-shift column, 
        # alternatively could normalize A + BK below 
        a = (self.norm(self.a, ord=self.norm_order) 
             if self.norm_order > 0 else self.a)
        A = self.get_companion_matrix(a)
        
        if self.closed_loop:  # Compute closed-loop forecast
            # Compute hidden state 
            # -> x_lag = \sum_{i = 0}^{lag - 1} A^{lag - 1 - i}B u_i
            k_x = krylov_logarithmic_products(l, A, self.b, c=None)
            x = self.fft_conv_d(u, k_x)  # shape: B x H x L x D
            
            # Compute A + BK matrix
            b = (self.norm(self.b, ord=self.norm_order) 
                 if self.norm_order > 0 else self.b)
            k = (self.norm(self.k, ord=self.norm_order) 
                 if self.norm_order > 0 else self.b)

            A_BK = A + oe.contract('h i, h j -> h i j', b, k)
            
            # Rollout: Compute C(A + BK)^{h} * x_lag and K(A + BK)^{h} * x_lag
            # First compute hidden state
            x = krylov_logarithmic_products(l_horizon, A_BK, x[:, :, -1, :], c=None)
            
            # Compute predictions for layer output
            c = self.norm(self.c, ord=self.norm_order) if self.norm_order > 0 else self.c
            y = torch.einsum('...nl, ...n -> ...l', x, c).contiguous()
            y = y.transpose(1, 2)
            
            # Compute predictions for layer next-time-step input (prior layer next-time-step output)
            if not self.inference_only and self.closed_loop:
                u = torch.einsum('...nl, ...n -> ...l', x, self.k).contiguous()
                u = u.transpose(1, 2)
            else:
                u = None
            
            # Layer outputs, and next-time-step layer inputs
            return y, u
        
        else:  # Compute open-loop forecast up to L
            # A = self.norm(A, ord=self.norm_order)
            # Return CA^{n}B where A = a is computed companion matrix from self.a
            b = (self.norm(self.b, ord=self.norm_order) 
                 if self.norm_order > 0 else self.b)
            c = self.norm(self.c, ord=self.norm_order) if self.norm_order > 0 else self.c
            k = krylov_logarithmic_products(l, A, b, c)
            k = repeat(k, 'nk kd -> (kr nk nh hd) kd', 
                       kr=self.kernel_repeat, nh=self.num_heads, hd=self.head_dim)
            y = self.fft_conv(u, k).transpose(1, 2)
            
            if not self.inference_only:
                _k  = self.norm(self.k, ord=self.norm_order)
                k_u = krylov_logarithmic_products(l, A, b, _k)
                k_u = repeat(k_u, 'nk kd -> (kr nk nh hd) kd', 
                             kr=self.kernel_repeat, nh=self.num_heads, hd=self.head_dim)
                y_u = self.fft_conv(u, k_u).transpose(1, 2)
            else:
                y_u = None
            return y, (y_u, u.transpose(1, 2))

