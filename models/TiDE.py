import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPResidual(nn.Module):
    """
    MLPResidual
    """

    def __init__(self, input_dim, hidden_size, output_dim, dropout, layernorm):
        super().__init__()
        self.layernorm = layernorm
        if layernorm:
            self.norm = nn.LayerNorm(output_dim)

        self.drop = nn.Dropout(dropout)
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_dim)
        self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        # MLP dense
        x = F.relu(self.lin1(input))
        x = self.lin2(x)
        x = self.drop(x)

        # Skip connection
        x_skip = self.skip(input)

        # Combine
        x = x + x_skip

        if self.layernorm:
            return self.norm(x)

        return x

class Model(nn.Module):
    """TiDE

    Time-series Dense Encoder (`TiDE`) is a MLP-based univariate time-series forecasting model. `TiDE` uses Multi-layer Perceptrons (MLPs) in an encoder-decoder model for long-term time-series forecasting.

    **Parameters:**<br>
    `h`: int, forecast horizon.<br>
    `input_size`: int, considered autorregresive inputs (lags), y=[1,2,3,4] input_size=2 -> lags=[1,2].<br>
    `hidden_size`: int=1024, number of units for the dense MLPs.<br>
    `decoder_output_dim`: int=32, number of units for the output of the decoder.<br>
    `temporal_decoder_dim`: int=128, number of units for the hidden sizeof the temporal decoder.<br>
    `dropout`: float=0.0, dropout rate between (0, 1) .<br>
    `layernorm`: bool=True, if True uses Layer Normalization on the MLP residual block outputs.<br>
    `num_encoder_layers`: int=1, number of encoder layers.<br>
    `num_decoder_layers`: int=1, number of decoder layers.<br>
    `temporal_width`: int=4, lower temporal projected dimension.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `exclude_insample_y`: bool=False, whether to exclude the target variable from the historic exogenous data.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of different series in each batch.<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch.<br>
    `windows_batch_size`: int=1024, number of windows to sample in each training batch, default uses all.<br>
    `inference_windows_batch_size`: int=1024, number of windows to sample in each inference batch, -1 uses all.<br>
    `start_padding_enabled`: bool=False, if True, the model will pad the time series with zeros at the beginning, by input size.<br>
    `step_size`: int=1, step size between each window of temporal data.<br>
    `scaler_type`: str='identity', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int=1, random_seed for pytorch initializer and numpy generators.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `optimizer`: Subclass of 'torch.optim.Optimizer', optional, user specified optimizer instead of the default choice (Adam).<br>
    `optimizer_kwargs`: dict, optional, list of parameters used by the user specified `optimizer`.<br>
    `lr_scheduler`: Subclass of 'torch.optim.lr_scheduler.LRScheduler', optional, user specified lr_scheduler instead of the default choice (StepLR).<br>
    `lr_scheduler_kwargs`: dict, optional, list of parameters used by the user specified `lr_scheduler`.<br>
    `dataloader_kwargs`: dict, optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`. <br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>

    **References:**<br>
    - [Das, Abhimanyu, Weihao Kong, Andrew Leach, Shaan Mathur, Rajat Sen, and Rose Yu (2024). "Long-term Forecasting with TiDE: Time-series Dense Encoder."](http://arxiv.org/abs/2304.08424)

    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = False  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(self, config):

        # Inherit BaseWindows class
        super().__init__()

        self.c_in = 1 if config.features=='S' else 7
        self.h = config.pred_len
        input_size = config.seq_len * config.c_out

        hidden_size = config.d_model
        decoder_output_dim = config.label_len
        temporal_decoder_dim = 128
        dropout = 0.3
        layernorm = True
        num_encoder_layers = config.enc_in
        num_decoder_layers = config.dec_in
        temporal_width = config.factor
        
        self.hist_exog_size = 0
        self.futr_exog_size = 0
        self.stat_exog_size = 0

        # Encoder
        dense_encoder_input_size = (
            input_size
            + input_size * (self.hist_exog_size > 0) * temporal_width
            + (input_size + self.h) * (self.futr_exog_size > 0) * temporal_width
            + (self.stat_exog_size > 0) * self.stat_exog_size
        )

        dense_encoder_layers = [
            MLPResidual(
                input_dim=dense_encoder_input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                output_dim=hidden_size,
                dropout=dropout,
                layernorm=layernorm,
            )
            for i in range(num_encoder_layers)
        ]
        self.dense_encoder = nn.Sequential(*dense_encoder_layers)

        # Decoder
        decoder_output_size = decoder_output_dim * self.h
        dense_decoder_layers = [
            MLPResidual(
                input_dim=hidden_size,
                hidden_size=hidden_size,
                output_dim=(
                    decoder_output_size if i == num_decoder_layers - 1 else hidden_size
                ),
                dropout=dropout,
                layernorm=layernorm,
            )
            for i in range(num_decoder_layers)
        ]
        self.dense_decoder = nn.Sequential(*dense_decoder_layers)

        # Temporal decoder with loss dependent dimensions
        self.temporal_decoder = MLPResidual(
            input_dim=decoder_output_dim + (self.futr_exog_size > 0) * temporal_width,
            hidden_size=temporal_decoder_dim,
            output_dim=self.c_in,
            dropout=dropout,
            layernorm=layernorm,
        )

        # Global skip connection
        self.global_skip = nn.Linear(
            in_features=input_size, out_features=self.h * self.c_in
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Parse windows_batch
        x = x_enc  #   [B, L, 1]
        hist_exog = []  #   [B, L, X]
        futr_exog = []  #   [B, L + h, F]
        stat_exog = []  #   [B, S]
        batch_size, seq_len = x.shape[:2]  #   B = batch_size, L = seq_len

        # Flatten insample_y
        x = x.reshape(batch_size, -1)  #   [B, L, 1] -> [B, L]

        # Global skip connection
        x_skip = self.global_skip(x)  #   [B, L] -> [B, h * n_outputs]
        x_skip = x_skip.reshape(
            batch_size, self.h, -1
        )  #   [B, h * n_outputs] -> [B, h, n_outputs]

        # Dense encoder
        x = self.dense_encoder(
            x
        )  #   [B, L * (1 + 2 * temporal_width) + h * temporal_width + S] -> [B, hidden_size]

        # Dense decoder
        x = self.dense_decoder(x)  #   [B, hidden_size] ->  [B, decoder_output_dim * h]
        x = x.reshape(
            batch_size, self.h, -1
        )  #   [B, decoder_output_dim * h] -> [B, h, decoder_output_dim]

        # Stack with futr_exog for horizon part of futr_exog
        if self.futr_exog_size > 0:
            x_futr_exog_h = x_futr_exog[
                :, seq_len:
            ]  #  [B, L + h, temporal_width] -> [B, h, temporal_width]
            x = torch.cat(
                (x, x_futr_exog_h), dim=2
            )  #  [B, h, decoder_output_dim] + [B, h, temporal_width] -> [B, h, temporal_width + decoder_output_dim]

        # Temporal decoder
        x = self.temporal_decoder(
            x
        )  #  [B, h, temporal_width + decoder_output_dim] -> [B, h, n_outputs]

        forecast = x + x_skip

        return forecast
