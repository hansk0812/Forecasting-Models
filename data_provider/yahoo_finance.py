import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Yahoo_Finance(Dataset):
    
    # Implements multi-variate forecasting over 12 S&P 500 stocks
    STOCKS = ["AAPL", "MCD", "ABT", "MSFT", "AEM", "ORCL",
              "AFG", "WWD", "APA", "T", "CAT", "RTX"]
    SPLITS = [0.7, 0.15, 0.15] # train, val, test

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=None,
                 target='MSFT', scale=True, timeenc=0, freq='d', cycle=32):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        fnames = [os.path.join(self.root_path, stock_name + ".csv") for stock_name in self.STOCKS]
        df_raw = [None for _ in range(len(fnames))]

        for idx, fn in enumerate(fnames):
            df_raw[idx] = pd.read_csv(fn).iloc[2:].reset_index(drop=True)
            if idx == 0:
                date_cols = df_raw[idx][["Price"]].rename(columns={"Price": "date"})
            df_raw[idx] = df_raw[idx][["Close"]].rename(columns={"Close": self.STOCKS[idx]})

        df_raw = pd.concat([date_cols] + df_raw, axis=1)
        
        trainable = len(df_raw) - self.seq_len - self.pred_len + 1
        train_end = int(self.SPLITS[0] * trainable)
        val_end = train_end + int(self.SPLITS[1] * trainable)
        test_end = trainable
        border1s = [0, train_end, val_end]
        border2s = [train_end + self.seq_len + self.pred_len - 1, 
                    val_end + self.seq_len + self.pred_len - 1, 
                    test_end + self.seq_len + self.pred_len - 1]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            nf=1
        elif self.features == 'SM':
            cols_data = df_raw.columns[1:]
            df_data = pd.concat([df_raw[[c]].rename(columns={c:"M"}) for c in cols_data], axis=0).sort_index().reset_index(drop=True)
            nf=len(cols_data)
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            nf=1

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date, format="%Y-%m-%d")
        self.freq_diff = df_stamp['date'].iloc[1] - df_stamp['date'].iloc[0]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1*nf:border2*nf]
        
        print ("Dataset total number of timesteps: %d" % len(data))
        print ("Dataset %d length: %d" % (self.set_type, len(self)))
 
    def get_num_features(self):
        return len(self.STOCKS)

    def frequency(self):
        return self.freq_diff

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle = self.cycle_index[s_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle

    def __len__(self):
        return len(self.data_x) - self.pred_len - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
