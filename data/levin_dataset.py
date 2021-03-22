'''
levin dataset
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import cv2
import glob
try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger('base')


class LevinDataset(data.Dataset):
    def __init__(self, opt):
        super(LevinDataset, self).__init__()
        self.opt = opt
        # temporal augmentation

        HQ_pattern = opt['HQ_pattern']
        LQ_pattern = opt['LQ_pattern']
        self.N_frames = opt['N_frames']

        self.paths_HQ = sorted(glob.glob(HQ_pattern))
        self.paths_LQ = sorted(glob.glob(LQ_pattern))

    def __getitem__(self, index):
        HQ_size = self.opt['HQ_size']
        #### get the images
        HQ_path = self.paths_HQ[index]
        LQ_path = self.paths_LQ[index]
        img_HQ = cv2.imread(HQ_path, cv2.IMREAD_GRAYSCALE) / 255.
        img_LQ = cv2.imread(LQ_path, cv2.IMREAD_GRAYSCALE) / 255.

        img_HQ = np.expand_dims(img_HQ, axis=2)
        img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt['phase'] == 'train':
            _, H, W = 3, 256, 256  # LQ size
            # randomly crop
            rnd_h = random.randint(0, max(0, H - HQ_size))
            rnd_w = random.randint(0, max(0, W - HQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + HQ_size, rnd_w:rnd_w + HQ_size, :]
            img_HQ = img_HQ[rnd_h:rnd_h + HQ_size, rnd_w:rnd_w + HQ_size, :]

            # augmentation - flip, rotate
            rlt = util.augment([img_HQ, img_LQ], self.opt['use_flip'], self.opt['use_rot'])
            img_HQ = rlt[0]
            img_LQ = rlt[1]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQ = np.transpose(img_LQ, (2, 0, 1))
        img_HQ = np.transpose(img_HQ, (2, 0, 1))
        img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ)).float()
        img_HQ = torch.from_numpy(np.ascontiguousarray(img_HQ)).float()

        img_LQ = img_LQ
        img_HQ = img_HQ.unsqueeze(0)

        return {'LQ': img_LQ, 'HQs': img_HQ}

    def __len__(self):
        return len(self.paths_HQ)
