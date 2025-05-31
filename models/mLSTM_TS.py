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

def create_xlstm_model(config):
    # Define your input size, hidden size, and other relevant parameters
    input_size = 1 if config.features=='S' else 7  # Number of features in your time series
    embedding_dim = config.d_model #64  # Dimension of the embeddings, reduced to save memory
    output_size = input_size  # Number of output features (predicting the next value)
    seq_length = config.seq_len

    # Define the xLSTM configuration
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=config.enc_in, qkv_proj_blocksize=config.enc_in//2, num_heads=config.n_heads//2  # enc_in=4, n_heads=4 Reduced parameters to save memory
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=config.n_heads,  # 4 Reduced number of heads to save memory
                conv1d_kernel_size=config.factor,  # 2 Reduced kernel size to save memory
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.1, act_fn="gelu"),  # Reduced projection factor to save memory
        ),
        context_length=seq_length,
        num_blocks=config.e_layers,  # Reduced number of blocks to save memory
        embedding_dim=embedding_dim,
        slstm_at=[1 if config.e_layers>1 else 0],
    )

    # Instantiate the xLSTM stack
    xlstm_stack = xLSTMBlockStack(cfg)

    # Add a linear layer to project input data to the required embedding dimension
    input_projection = nn.Linear(input_size, embedding_dim)

    # Add a final linear layer to project the xLSTM output to the desired output size
    output_projection = nn.Linear(embedding_dim, output_size)

    return xlstm_stack, input_projection, output_projection

# -------------------------------------------------------------------------------------------
# Plot architecture
# -------------------------------------------------------------------------------------------

# Define a simplified model to pass through torchinfo
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.xlstm_stack, self.input_projection, self.output_projection = create_xlstm_model(config)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x = self.input_projection(x_enc)
        x = self.xlstm_stack(x)
        x = self.output_projection(x)
        return x
