import pandas as pd
from sklearn.preprocessing import StandardScaler

import numpy as np
import datetime

from torch.utils.data import Dataset

import os

from utils.timefeatures import time_features

class ExchangeDataset(Dataset):

    def __init__(self, root_path, data_path, flag, size, features, cycle, target="Singapore", timeenc=0, freq='h', scale="instance"):
        
        # flag: train, val, test, pred
        # size: lag, label_len, horizon
        # features: S, M, SM
        # target: Singapore
        # timeenc = 0 if args.embed != 'timeF' else 1
        # freq: h
        
        file_path = os.path.join(root_path, data_path)
        num_variates = 8 if 'M' in features else 1
        lag, label_len, horizon = size

        data = pd.read_csv(file_path, sep=',', skiprows=1)
        # Exchange dataset from 1990 -> 2016 daily over weekdays: Change weekly_flag to have daily data regardless
        weekly_flag = False

        dts = [[datetime.datetime.strptime("01/01/1990", "%d/%m/%Y")]] # Monday
        diffs_delta = datetime.timedelta(days=1)
        weekly_index = 1
        for _ in range(len(data) - 1):
            if weekly_index == 5 and weekly_flag:
                diffs_delta = datetime.timedelta(days=3)
                weekly_index = 0
            elif weekly_index == 1 and weekly_flag:
                diffs_delta = datetime.timedelta(days=1)
            dts.append([dts[-1][0] + diffs_delta])
            weekly_index += 1

        data_len = len(data)
        # Remove rows without data recorded where all columns are 0
        data = data.loc[~(data==0).all(axis=1)]
        data = data.dropna()
        if len(data) != data_len:
            print ("Removed %d rows from dataset, dates might be affected!" % data_len - len(data))

        dates = pd.DataFrame(dts, columns=["date"])
        dates.date = pd.to_datetime(dates["date"])

        if timeenc == 0:
            dates['month'] = dates.date.apply(lambda row: row.month, 1)
            dates['day'] = dates.date.apply(lambda row: row.day, 1)
            dates['weekday'] = dates.date.apply(lambda row: row.weekday(), 1)
            dates['hour'] = dates.date.apply(lambda row: row.hour, 1)
            data_stamp = dates.drop(['date'], axis=1).values
        elif timeenc == 1:
            data_stamp = time_features(pd.to_datetime(dates['date'].values), freq=freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

        if features == "SM":
            cols_data = data.columns
            data = pd.concat([data[[c]].rename(columns={c:"M"}) for c in cols_data], axis=0).sort_index().reset_index(drop=True)
            self.data_stamp = np.tile(self.data_stamp, (len(cols_data), 1))
            #nf=len(cols_data)

        self.cycle_index = (np.arange(len(data)) % cycle)

        n = len(data) - horizon - lag + 1
        save_stats_file = os.path.join(root_path, data_path.split('.')[0] + "_%s_numfeatures%d.npy" % (
                                        features, num_variates))

        if not os.path.exists(save_stats_file):
            scaler = StandardScaler()
            scaler.fit(data[:np.ceil(0.6*n).astype(np.int32)])
            self.mean, self.std = scaler.mean_, scaler.var_**0.5
            with open(save_stats_file, "wb") as f:
                np.savez(f, self.mean, self.std)
        else:
            if scale == "zscore":
                with open(save_stats_file, "rb") as f:
                    stats = np.load(f)
                    self.mean = np.array(stats["arr_0"])
                    self.std = np.array(stats["arr_1"])

        train_last = int(0.6*n)
        val_last = train_last + int(0.2*n)
        if flag == "train":
            data = data.iloc[:train_last + horizon]
            self.cycle_index = self.cycle_index[:train_last + horizon]
            self.data_stamp = self.data_stamp[:train_last + horizon]
        elif flag == "val":
            data = data.iloc[train_last:val_last + horizon]
            self.cycle_index = self.cycle_index[train_last:val_last + horizon]
            self.data_stamp = self.data_stamp[train_last:val_last + horizon]
        else:
            data = data.iloc[int(n*0.8):]
            self.cycle_index = self.cycle_index[val_last:]
            self.data_stamp = self.data_stamp[val_last:]
        
        if scale == "zscore":
            data_scaled = (data - self.mean) / (self.std + 1e-7) #scaler.transform(data)
            data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
        else:
            data_scaled = data
        self.data = np.array(data_scaled)
        self.data_stamp = np.array(self.data_stamp)
        
        self.instance_norm = (scale == "instance")

        self.pred_len = horizon
        self.seq_len = lag
        self.label_len = label_len

        self.num_features = num_variates

    def get_num_features(self):
        return self.num_features

    def __len__(self):
        return len(self.data) - self.pred_len - self.seq_len + 1

    def __getitem__(self, index):
        
        if index >= len(self):
            raise StopIteration 

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = self.cycle_index[s_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

       
if __name__ == "__main__":
    
    dataset = ExchangeDataset("dataset/Exchange/", "exchange_rate.txt.gz", "train", [720,360,720], 'M', cycle=64, target="Australia", timeenc=0, freq='h', scale="instance")
    print (len(dataset))

    from torch.utils.data import DataLoader
    dataset = DataLoader(dataset, batch_size=100) #, shuffle=True)

    for idx, (x, y, xm, ym, c) in enumerate(dataset):
        print (x.shape, y.shape, xm.shape, ym.shape, x.min(), x.max(), y.min(), y.max())

    dataset = ExchangeDataset("dataset/Exchange/", "exchange_rate.txt.gz", "val", [720,360,720], 'M', cycle=64, target="Australia", timeenc=0, freq='h', scale="instance")
    print (len(dataset))

    from torch.utils.data import DataLoader
    dataset = DataLoader(dataset, batch_size=100) #, shuffle=True)

    for idx, (x, y, xm, ym, c) in enumerate(dataset):
        print (x.shape, y.shape, xm.shape, ym.shape, x.min(), x.max(), y.min(), y.max())

    dataset = ExchangeDataset("dataset/Exchange/", "exchange_rate.txt.gz", "test", [720,360,720], 'M', cycle=64, target="Australia", timeenc=0, freq='h', scale="instance")
    print (len(dataset))

    from torch.utils.data import DataLoader
    dataset = DataLoader(dataset, batch_size=100) #, shuffle=True)

    for idx, (x, y, xm, ym, c) in enumerate(dataset):
        print (x.shape, y.shape, xm.shape, ym.shape, x.min(), x.max(), y.min(), y.max())
