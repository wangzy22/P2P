import os
import numpy as np
import warnings
import pickle

import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNet(Dataset):
    def __init__(self, config, split):
        self.root = config.data_root
        self.use_normals = config.use_normals
        self.num_category = config.classes
        self.split = split

        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.shape_ids = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_{}.txt'.format(split)))]
        print('The size of %s data is %d' % (split, len(self.shape_ids)))

        self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, 8192))
        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.shape_ids)

    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]
            point_set = point_set[:, [2, 0, 1]] * np.array([[-1, -1, 1]])
        else:
            point_set[:, :3] = point_set[:, [2, 0, 1]] * np.array([[-1, -1, 1]])
            point_set[:, 3:] = point_set[:, [5, 3, 4]] * np.array([[-1, -1, 1]])
        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])
        if self.split == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return current_points, label
