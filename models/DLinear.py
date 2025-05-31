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

        self.c_out = 1 if config.features=='S' else 7
        self.output_attention = False

        # Decomposition
        self.decomp = SeriesDecomp(moving_avg_window)

        self.linear_trend = nn.Linear(
            self.input_size * self.c_out, self.c_out * self.h, bias=True
        )
        self.linear_season = nn.Linear(
            self.input_size * self.c_out, self.c_out * self.h, bias=True
        )

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
