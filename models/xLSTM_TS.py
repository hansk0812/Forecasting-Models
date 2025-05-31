# src/ml/models/xlstm_ts/xlstm_ts_model.py

# -------------------------------------------------------------------------------------------
# New proposed model: xLSTM-TS, a time series-specific implementation
# 
# References:
# 
# - Paper (2024): https://doi.org/10.48550/arXiv.2405.04517
# - Official code: https://github.com/NX-AI/xlstm
# - Parameters for time series: https://github.com/smvorwerk/xlstm-cuda
# -------------------------------------------------------------------------------------------

import torch

import torch.nn as nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

RECURRENT_FACTOR = 48

torch.autograd.set_detect_anomaly(True)

def create_xlstm_model(config):
    K = 5
    # Define your input size, hidden size, and other relevant parameters
    input_size = 1 if config.features=='S' else 7  # Number of features in your time series
    embedding_dim = config.factor * K if not config.recurrent else config.d_model #RECURRENT_FACTOR * input_size #64  # Dimension of the embeddings, reduced to save memory
    output_size = input_size  # Number of output features (predicting the next value)
    seq_length = config.seq_len
    
    print ("dropout:", config.dropout)
    # Define the xLSTM configuration
    mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=config.factor, qkv_proj_blocksize=config.enc_in, num_heads=config.n_heads//2,  # nheads//2 enc_in=4, n_heads=4 Reduced parameters to save memory
                bias=True, dropout=config.dropout,
                #channel_mixing=False,
                embedding_dim=config.factor*K if not config.recurrent else config.d_model, #RECURRENT_FACTOR*input_size,
                proj_factor=2.,
                )
            )
    slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=config.n_heads,  # 4 Reduced number of heads to save memory
                conv1d_kernel_size=config.factor,  # 2 Reduced kernel size to save memory
                bias_init="powerlaw_blockdependent",
                dropout=config.dropout,
                #channel_mixing=False,
                #embedding_dim=config.factor*8, # hardcoded gyan
                ),
                feedforward=FeedForwardConfig(proj_factor=1.1, act_fn="gelu"), # Reduced projection factor to save memory
            )
    
    if not config.recurrent:
        cfg = xLSTMBlockStackConfig(
            slstm_block=slstm_block,
            mlstm_block=mlstm_block,
            context_length=seq_length,
            num_blocks=config.e_layers,  # Reduced number of blocks to save memory
            #embedding_dim=config.factor * 8, # hardcoded gyan
            slstm_at=[1 if config.e_layers>1 else 0],
            add_post_blocks_norm=False,
        )
    else:
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mlstm_block,
            context_length=seq_length,
            num_blocks=config.e_layers,  # Reduced number of blocks to save memory
            #embedding_dim=config.factor * 8, # hardcoded gyan
            add_post_blocks_norm=False,
        )
    
    # Instantiate the xLSTM stack
    xlstm_stack = xLSTMBlockStack(cfg)

    # Add a linear layer to project input data to the required embedding dimension
    if not config.recurrent:
        input_projection = nn.Linear(input_size, embedding_dim) #input_size)
    else:
        input_projection = nn.Linear(input_size*(config.seq_len//RECURRENT_FACTOR), embedding_dim) #input_size)

    # Add a final linear layer to project the xLSTM output to the desired output size
    output_projection = nn.Linear(embedding_dim, output_size*(config.seq_len//RECURRENT_FACTOR) if config.recurrent else output_size) #output_size, output_size)

    return xlstm_stack, input_projection, output_projection

# -------------------------------------------------------------------------------------------
# Plot architecture
# -------------------------------------------------------------------------------------------

# Define a simplified model to pass through torchinfo
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.xlstm_stack, self.input_projection, self.output_projection = create_xlstm_model(config)
        self.recurrent = config.recurrent

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        if not self.recurrent:
            x = self.input_projection(x_enc)
            x = self.xlstm_stack(x)
            out = self.output_projection(x)
        else:
            patch_len, state = x_enc.shape[-2]//RECURRENT_FACTOR, None
            num_series = x_enc.shape[-1]
            out = torch.zeros_like(x_enc)
            for idx in torch.arange(0, x_enc.shape[-2], patch_len):
                x_input = x_enc[:,idx:idx+patch_len,:].view((-1,1,patch_len*num_series))
                x_input = self.input_projection(x_input)
                y_step, state = self.xlstm_stack.step(x_input, state)
                y_step = self.output_projection(y_step)
                out[:,idx:idx+patch_len,:] = y_step.view((y_step.shape[0],patch_len,-1))
        return out
