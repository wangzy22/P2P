import os
import h5py
import numpy as np

import torch


class ScanObjectNN(torch.utils.data.Dataset):
    def __init__(self, config, subset):
        super().__init__()
        self.root = config.data_root
        self.subset = subset
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()
        current_points = current_points[:, [2, 0, 1]] * np.array([[-1, -1, 1]])
    
        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        
        return current_points, label

    def __len__(self):
        return self.points.shape[0]