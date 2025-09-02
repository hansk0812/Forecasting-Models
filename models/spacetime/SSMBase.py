# https://arxiv.org/pdf/2303.09489

import torch
import torch.nn as nn

import einops
import opt_einsum as oe

import traceback


def krylov_logarithmic_products(N, A, b, c=None):
    
    assert N>0, "Positive powers only"

    Abcs = b.unsqueeze(-1)
    
    A_squares = A

    while Abcs.shape[-1] <= N - Abcs.shape[-1]:
        
        A_n_by_2_b = A_squares @ Abcs
        A_squares = A_squares @ A_squares
        Abcs = torch.cat([Abcs, A_n_by_2_b], dim=-1)
    
    A_n_by_2_b = A_squares @ Abcs
    Abcs = torch.cat([Abcs, A_n_by_2_b], dim=-1)
    Abcs = Abcs[...,:N]

    if not c is None:
        Abcs = torch.einsum('...nl, ...n -> ...l', Abcs, c)
    Abcs = Abcs.contiguous() # WOW!!
    
    return Abcs

class FFN(nn.Module):

    def __init__(self, input_size, output_size):

        super().__init__()

        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()

        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.gelu2(self.linear(self.gelu1(x)))

class SSMBase(nn.Module):

    def __init__(self, model_dim, num_kernels, kernel_dim, kernel_repeat, num_heads, kernel_weights, kernel_init, kernel_train, skip_connection, seed, head_dim=None):

        super().__init__()

        self.model_dim = model_dim
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim
        self.kernel_repeat = kernel_repeat

        self.num_heads = num_heads
        self.head_dim = head_dim if not head_dim is None else self.model_dim // (self.kernel_repeat * self.num_kernels * num_heads)

        self.kernel_weights = kernel_weights
        self.skip_connection = skip_connection
        
        self.kernel_init     = kernel_init
        self.kernel_train    = kernel_train
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        self.init_weights()
    
    def init_weights(self):
        if not self.kernel_weights is None:  
            self.register('k', self.kernel_weights, trainable=True)
        
        skip = torch.randn(self.model_dim)
        self.register('skip', skip, trainable=True)
 
    def fft_conv(self, x, k):
        L   = x.shape[-1]
        x_f = torch.fft.rfft(x, n=2*L) # (B H L)
        k_f = torch.fft.rfft(k[:, :L], n=2*L) # (H L)
        
        y_f = x_f * k_f
        #y_f = oe.contract('b h l, h l -> b h l', x, k)
        #y = y_f
        #y_f = oe.contract('b h l, h l -> b h l', x_f, k_f)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L]  # (B H L)
        return y

    def register(self, name, tensor, trainable):
        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)
    
    def get_kernel(self):
        raise NotImplementedError

    def forward(self, x):
        x = x.transpose(1, 2)
        
        # Repeat kernels across heads
        if self.kernel_weights is None:
            k = self.get_kernel(x)
            k = einops.repeat(k, 'nk kd -> (kr nk nh hd) kd',
                   kr=self.kernel_repeat, nh=self.num_heads, hd=self.head_dim)
        else:
            k = self.k
        
        try:
            y = self.fft_conv(x, k)
        except Exception as e:
            print(e)
            traceback.print_exc()
            breakpoint()
        
        if self.skip_connection:
            y = y + oe.contract('b d l, d -> b d l', x, self.skip)
        y = y.transpose(1, 2)
        return y

