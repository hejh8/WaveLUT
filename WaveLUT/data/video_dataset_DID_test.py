import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import torch.nn.functional as F
import random
import cv2
import numpy as np
import glob
import os


class Video_Dataset_DID_test(data.Dataset):
    def __init__(self, conf):
        super(Video_Dataset_DID_test, self).__init__()
        self.conf = conf
        self.cache_data = conf['cache_data']
        self.GT_root, self.LQ_root = conf['dataroot_GT'], conf['dataroot_LQ']
        self.data_type = self.conf['data_type']
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        self.folders = []
        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = [], []

        # read data:
        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)
        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            subfolder_name = osp.basename(subfolder_GT)

            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT = util.glob_file_list(subfolder_GT)

            max_idx = len(img_paths_LQ) # default 7
            assert max_idx == len(img_paths_GT), 'Different number of images in LQ and GT folders'
            self.folders.append(subfolder_name)

            if self.cache_data:
                self.imgs_LQ.append(img_paths_LQ)
                self.imgs_GT.append(img_paths_GT)

    def __getitem__(self, index):
        imgs_LQ_paths = self.imgs_LQ[index]
        imgs_GT_paths = self.imgs_GT[index]

        imgs_LQ = util.read_img_seq2(imgs_LQ_paths, self.conf['train_size'])
        imgs_GT = util.read_img_seq2(imgs_GT_paths, self.conf['train_size'])

        img_LQ_l = list(imgs_LQ.unbind(0))
        img_GT_l = list(imgs_GT.unbind(0))

        return {
            'LQs': torch.stack(img_LQ_l).permute(1, 0, 2, 3).contiguous(),  # shape: [C, N, H, W]
            'GTs': torch.stack(img_GT_l).permute(1, 0, 2, 3).contiguous(),  # shape: [C, N, H, W]
            'folder': self.folders[index],
        }

    def __len__(self):
        return len(self.folders)
