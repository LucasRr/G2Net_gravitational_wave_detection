
import numpy as np
import pandas as pd
import os
import h5py

import torch
from torch.utils.data import DataLoader

class G2NetDataset(torch.utils.data.Dataset):

    def __init__(self, path, dataset_name, csv_name=None, normalize=False, augment=False, use_labels=True):
        self.path = path
        self.dataset_name = dataset_name
        if csv_name is None:
            csv_name = dataset_name + "_labels.csv"
        df = pd.read_csv(os.path.join(path, csv_name), dtype={"id":str})
        df = df[df.target >= 0]  # Remove 3 unknowns (target = -1)
        self.df = df
        self.normalize = normalize
        self.augment = augment
        self.use_labels = use_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]  # i-th column

        file_id = row.id
        if self.use_labels:
            y = np.float32(row.target)
        else:
            y = np.float32(0.5)  # ignore this label
        
        filename = os.path.join(self.path, self.dataset_name, file_id+".hdf5")

        with h5py.File(filename, 'r') as f:
            H1 = f[file_id]['H1']
            L1 = f[file_id]['L1']
            H1_SFTs = np.array(H1['SFTs'])[:,:4272] * 1e22  # 4272/48 = 89
            L1_SFTs = np.array(L1['SFTs'])[:,:4272] * 1e22
        
            H1_pow = np.abs(H1_SFTs)**2
            L1_pow = np.abs(L1_SFTs)**2

            if self.augment:
                # randomly set columns to zeros:
                num_gaps = 10
                max_gap_length = 30

                gap_start_idx = np.random.randint(0, 4272-max_gap_length, num_gaps)  # 10 gaps
                gap_channel = np.random.randint(0, 2, num_gaps)
                gap_length = np.random.randint(1, max_gap_length, (num_gaps,))

                for i, j, l in zip(gap_channel, gap_start_idx, gap_length):
                    if i == 0:
                        H1_pow[:,j:(j+l)] = 0.0
                    else:
                        L1_pow[:,j:(j+l)] = 0.0

            if self.normalize:
                mean_H1, var_H1 = np.mean(H1_pow), np.var(H1_pow)
                mean_L1, var_L1 = np.mean(L1_pow), np.var(L1_pow)
                
                H1_pow = (H1_pow - mean_H1)/np.sqrt(var_H1)
                L1_pow = (L1_pow - mean_L1)/np.sqrt(var_L1)
                
            x = np.stack((H1_pow, L1_pow), axis=0)
            
            x = torch.from_numpy(x).type(torch.FloatTensor)

        if self.augment:
            # frequency shift:
            freq_shift = np.random.randint(0, 360)
            x = torch.concatenate((x[:, freq_shift:], x[:, :freq_shift]), axis=1)

        return x, y

        