import os

import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.MovingAvg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.MovingAvg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):

    def __init__(
        self,
        config):
        
        super().__init__()

        self.h = config.pred_len
        self.input_size = config.seq_len
        
        moving_avg_window = config.factor
        # Architecture
        if moving_avg_window % 2 == 0:
            raise Exception("moving_avg_window should be uneven")

        self.c_out = 1 if config.features=='S' else int(config.enc_in)
        self.output_attention = False

        # Decomposition
        self.decomp = SeriesDecomp(moving_avg_window)
        
        if (config.e_layers + config.d_layers) == 1:

            chkpt_path = glob.glob(os.path.join("checkpoints", "%s_%s_%s_modes%d_%s_ft%s_sl%d_ll%d_pl%d_dm%d_nh%d_el%d_dl%d_df%d_fc%d_eb%s_dt%s_%s_0" % (
        config.task_id.split("0.")[0] + '*', config.model, config.mode_select, config.modes, config.data, config.features, config.seq_len, config.label_len,
        config.pred_len, config.d_model, config.n_heads, config.e_layers, config.d_layers, 
        config.d_ff, config.factor, config.embed, config.distil, config.des.split("0.")[0] + '*')))
            if len(chkpt_path) > 0:
                print ("Checkpoint %s exists! Exiting!" % chkpt_path)
                exit()

            self.linear_trend = nn.Linear(
                self.input_size * self.c_out, self.c_out * self.h, bias=True
            )
            self.linear_season = nn.Linear(
                self.input_size * self.c_out, self.c_out * self.h, bias=True
            )
        else:
            trend, season = [], []
            for idx in range(config.e_layers):
                if idx == 0:
                    trend.extend([nn.Linear(
                        self.input_size * self.c_out, config.d_model, bias=True),
                                  nn.Dropout(config.dropout)])
                    season.extend([nn.Linear(
                           self.input_size * self.c_out, config.d_model, bias=True),
                                  nn.Dropout(config.dropout)])
                else:
                    trend.extend([nn.Linear(
                        config.d_model, config.d_model, bias=True),
                                  nn.Dropout(config.dropout)])
                    season.extend([nn.Linear(
                        config.d_model, config.d_model, bias=True),
                                  nn.Dropout(config.dropout)])

            for idx in range(config.d_layers):
                if idx == config.d_layers - 1:
                    trend.append(nn.Linear(config.d_model, self.c_out * self.h, bias=True))
                    season.append(nn.Linear(config.d_model, self.c_out * self.h, bias=True))
                else:
                    trend.extend([nn.Linear(config.d_model, config.d_model, bias=True),
                                  nn.Dropout(config.dropout)])
                    season.extend([nn.Linear(config.d_model, config.d_model, bias=True),
                                  nn.Dropout(config.dropout)])
            self.linear_trend = nn.Sequential(*trend)
            self.linear_season = nn.Sequential(*season)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Parse windows_batch
        insample_y = x_enc.reshape((x_enc.shape[0], -1))

        # Parse inputs
        batch_size = len(insample_y)
        seasonal_init, trend_init = self.decomp(insample_y)

        trend_part = self.linear_trend(trend_init)
        seasonal_part = self.linear_season(seasonal_init)

        # Final
        forecast = trend_part + seasonal_part
        forecast = forecast.reshape(batch_size, self.h, self.c_out)
        return forecast
