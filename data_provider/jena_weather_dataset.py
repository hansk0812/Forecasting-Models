import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

import glob

warnings.filterwarnings('ignore')


class Dataset_Weather(Dataset):
    
    TRAIN_VAL_TEST = (0.8, 0.1, 0.1)

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='1',
                 target="T (degC)", scale="zscore", timeenc=0, freq='t', cycle=32):
        # data_path: 1 - Beutenberg (includes CO2 and Photosynthetic Radiation: 9 features)
        # data_path: 2 - Beutenberg and Saaleaue (excludes solar radiation features: 7 features) 
        # plant and soil data and other features: Not Implemented

        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 6 * 4
            self.label_len = int(24 * 6 * 0.25)
            self.pred_len = 24 * 6 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        # Chosen without considerations in environmental sciences, change if necessary
        # If adding a column name that uses the encoding, please change the column name using ascii in folder_to_df()
        if features == 'M':
            if data_path == '1':
                self.vars = ["p (mbar)", "T (degC)", "sh (g/kg)", 
                             "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", 
                             "wd (deg)", "PAR (�mol/m�/s)"] #, "CO2 (ppm)"] --> NaN values

            elif data_path == '2':
                self.vars = ["p (mbar)", "T (degC)", "sh (g/kg)", 
                             "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", 
                             "wd (deg)"]
            else:
                raise NotImplementedError
        
        else:
            self.vars = [target]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def folder_to_df(self, csv_files):
        
        if "mpi_roof" in csv_files[0]:
            recent_idx = csv_files.index(os.path.join(self.root_path, "mpi_roof.csv"))
        else:
            recent_idx = csv_files.index(os.path.join(self.root_path, "mpi_saale.csv"))

        temp_fname = csv_files[-1]
        csv_files[-1] = csv_files[recent_idx]
        csv_files[recent_idx] = temp_fname
        
        csv_files = sorted(csv_files[:-1], key = lambda x: x.split('_')[-1].split('.')[0]) + [csv_files[-1]]
        
        df_raw = None
        for fname in csv_files:
            if df_raw is None:
                df_raw = pd.read_csv(fname, encoding="cp1250")
                if self.data_path == '1':
                    common_ascii = "PAR"
                    diff_encoding_column = [x for x in df_raw.columns if common_ascii in x][0]
                    var_idx = [idx for idx, v in enumerate(self.vars) if common_ascii in v][0]
                    self.vars[var_idx] = diff_encoding_column
            else:
                df_raw = pd.concat([df_raw, pd.read_csv(fname, encoding="cp1250")], ignore_index=True)
        
        return df_raw

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = self.folder_to_df(glob.glob(os.path.join(self.root_path, "mpi_roof*.csv")))
        dataset_length = len(df_raw) - self.seq_len - self.pred_len + 1
        train_indices = (0, int(self.TRAIN_VAL_TEST[0] * dataset_length))
        if self.set_type == 0:
            indices = train_indices
        elif self.set_type == 1:
            indices = (int(self.TRAIN_VAL_TEST[0] * dataset_length), \
                       int((self.TRAIN_VAL_TEST[0] + self.TRAIN_VAL_TEST[1]) * dataset_length))
        else:
            indices = (int((self.TRAIN_VAL_TEST[0] + self.TRAIN_VAL_TEST[1]) * dataset_length), \
                       dataset_length)
        
        if self.data_path == '2':
            df_raw2 = self.folder_to_df(glob.glob(os.path.join(self.root_path, "mpi_saale*.csv")))
            dataset_length = len(df_raw2)
            train_indices = [train_indices] + [(0, int(self.TRAIN_VAL_TEST[0] * dataset_length))]
            if self.set_type == 0:
                indices = train_indices
            elif self.set_type == 1:
                indices = [indices] + [(int(self.TRAIN_VAL_TEST[0] * dataset_length), \
                                        int((self.TRAIN_VAL_TEST[0] + self.TRAIN_VAL_TEST[1]) * dataset_length))]
            else:
                indices = [indices] + [(int((self.TRAIN_VAL_TEST[0] + self.TRAIN_VAL_TEST[1]) * dataset_length), \
                                        dataset_length)]

        if self.features == 'M' or self.features == 'MS':
            
            df_data = df_raw[self.vars]
            if self.data_path == '2':
                df_data2 = df_raw2[self.vars]
            nf=1
                
        elif self.features == 'SM':
            cols_data = self.vars
            df_data = pd.concat([df_raw[[c]].rename(columns={c:"M"}) for c in cols_data], axis=0).sort_index().reset_index(drop=True)
            if self.data_path == '2':
                df_data2 = pd.concat([df_raw2[[c]].rename(columns={c:"M"}) for c in cols_data], axis=0).sort_index().reset_index(drop=True)
            nf=len(cols_data)
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            if self.data_path == '2':
                df_data2 = df_raw2[[self.target]]
            nf=1
        
        if self.scale == "zscore":
            if not isinstance(train_indices[0], tuple):
                train_data = df_data[train_indices[0]*nf: train_indices[1]*nf]
            else:
                train_data = df_data[train_indices[0][0]*nf: train_indices[0][1]*nf]
                train_data = pd.concat([train_data, 
                               df_data2[train_indices[1][0]*nf: train_indices[1][1]*nf]], ignore_index=True)
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            if self.data_path == '2':
                data2 = self.scaler.transform(df_data2.values)
        else:
            data = df_data.values
            if self.data_path == '2':
                data2 = df_data2.values
        
        if not isinstance(indices[0], tuple):
            df_stamp = df_raw[['Date Time']][indices[0]*nf: indices[1]*nf]
        else:
            df_stamp = df_raw[['Date Time']][indices[0][0]*nf: indices[0][1]*nf]
            df_stamp2 = df_raw2[['Date Time']][indices[1][0]*nf: indices[1][1]*nf]

        if nf > 1:
            df_stamp = df_stamp.loc[df_stamp.index.repeat(nf)]
            if self.data_path == '2':
                df_stamp2 = df_stamp2.loc[df_stamp.index.repeat(nf)]

        df_stamp['Date Time'] = pd.to_datetime(df_stamp["Date Time"], format="%d.%m.%Y %H:%M:%S")
        df_stamp = df_stamp.rename(columns={"Date Time": "date"})
        if self.data_path == '2':
            df_stamp2['Date Time'] = pd.to_datetime(df_stamp2["Date Time"], format="%d.%m.%Y %H:%M:%S")
            df_stamp2 = df_stamp2.rename(columns={"Date Time": "date"})

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        if self.data_path == '2':
            if self.timeenc == 0:
                df_stamp2['month'] = df_stamp2.date.apply(lambda row: row.month, 1)
                df_stamp2['day'] = df_stamp2.date.apply(lambda row: row.day, 1)
                df_stamp2['weekday'] = df_stamp2.date.apply(lambda row: row.weekday(), 1)
                df_stamp2['hour'] = df_stamp2.date.apply(lambda row: row.hour, 1)
                df_stamp2['minute'] = df_stamp2.date.apply(lambda row: row.minute, 1)
                df_stamp2['minute'] = df_stamp2.minute.map(lambda x: x // 15)
                data_stamp2 = df_stamp2.drop(['date'], axis=1).values
            elif self.timeenc == 1:
                data_stamp2 = time_features(pd.to_datetime(df_stamp2['date'].values), freq=self.freq)
                data_stamp2 = data_stamp2.transpose(1, 0)
        
        if not isinstance(indices[0], tuple):
            self.data_x = data[indices[0]*nf:indices[1]*nf + self.pred_len]
            self.data_y = data[indices[0]*nf:indices[1]*nf + self.pred_len]
            self.data_stamp = data_stamp
            self.cycle_index = (np.arange(len(data)) % self.cycle)[indices[0]*nf:indices[1]*nf + self.pred_len]
        else:
            self.data_x = data[indices[0][0]*nf:indices[0][1]*nf + self.pred_len]
            self.data_y = data[indices[0][0]*nf:indices[0][1]*nf + self.pred_len]
            self.data_stamp = data_stamp
            self.cycle_index = (np.arange(len(data)) % self.cycle)[indices[0][0]*nf:indices[0][1]*nf + self.pred_len]
            
            self.data_x = np.concatenate([self.data_x, data2[indices[1][0]*nf:indices[1][1]*nf + self.pred_len]], axis=0)
            self.data_y = np.concatenate([self.data_y, data2[indices[1][0]*nf:indices[1][1]*nf + self.pred_len]], axis=0)
            self.data_stamp = np.concatenate([self.data_stamp, data_stamp2], axis=0)
            self.cycle_index = np.concatenate([self.cycle_index, 
                                (np.arange(len(data2)) % self.cycle)[indices[1][0]*nf:indices[1][1]*nf + self.pred_len]], axis=-1)
        
        print ("Dataset total number of timesteps: %d" % len(self.data_x))
        print ("Dataset length: %d" % len(self))
    
    def get_num_features(self):
        return len(self.vars)
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = self.cycle_index[s_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

