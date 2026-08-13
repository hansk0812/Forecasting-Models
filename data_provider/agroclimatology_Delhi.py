import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class Dataset_AG_Delhi(Dataset):
    
    SPLITS = [0.6, 0.2, 0.2] # train, val, test

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=None,
                 target='T2M', scale=True, timeenc=0, freq='d', cycle=32):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 6
            self.label_len = 24 * 6
            self.pred_len = 24 * 6
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

        fname = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(fname, skiprows=20)
        
        df_raw = df_raw.drop(columns=["YEAR", "DOY"])
        start_date = datetime.strptime("01/01/2015", "%d/%m/%Y")
        td = [start_date + timedelta(days=idx) for idx in range(df_raw.shape[0])]
        df_raw.insert(loc=0, column="date", value=pd.to_datetime(td))
        
        train_end = int(self.SPLITS[0] * len(df_raw)) + self.seq_len + self.pred_len - 1
        val_end = int((self.SPLITS[0] + self.SPLITS[1]) * len(df_raw)) + self.seq_len + self.pred_len - 1
        test_end = len(df_raw)
        border1s = [0, train_end - self.seq_len - self.pred_len + 1, val_end - self.seq_len - self.pred_len - 1]
        border2s = [train_end, val_end, test_end]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            nf=1
            self.num_features = len(cols_data)
        elif self.features == 'SM':
            cols_data = df_raw.columns[1:]
            df_data = pd.concat([df_raw[[c]].rename(columns={c:"M"}) for c in cols_data], axis=0).sort_index().reset_index(drop=True)
            nf=len(cols_data)
            self.num_features = 1
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            nf=1
            self.num_features = 1
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date, format="%Y-%m-%d")
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1*nf:border2*nf]
        
        print ("Dataset total number of timesteps: %d" % len(data))
        print ("Dataset length: %d" % len(self))
 
    def get_num_features(self):
        return self.num_features

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
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

