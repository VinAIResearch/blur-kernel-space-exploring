'''
fewshot dataset
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
import glob
try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger('base')


class FewShotDataset(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000_00000000
    HQ: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt):
        super(FewShotDataset, self).__init__()
        self.opt = opt

        self.HQ_pattern, self.LQ_pattern = opt['HQ_pattern'], opt['LQ_pattern']
        self.HQ_size = opt['HQ_size']

        self.HQ_paths = sorted(glob.glob(self.HQ_pattern))
        self.LQ_paths = sorted(glob.glob(self.LQ_pattern))

    def __getitem__(self, index):
        #### get the images
        img_HQ_path = self.HQ_paths[index]
        img_LQ_path = self.LQ_paths[index]

        img_HQ = util.read_img(None, img_HQ_path)
        img_LQ = util.read_img(None, img_LQ_path)

        # randomly crop
        H, W, _ = img_HQ.shape
        rnd_h = random.randint(0, max(0, H - self.HQ_size))
        rnd_w = random.randint(0, max(0, W - self.HQ_size))
        img_LQ = img_LQ[rnd_h:rnd_h + self.HQ_size, rnd_w:rnd_w + self.HQ_size, :]
        img_HQ = img_HQ[rnd_h:rnd_h + self.HQ_size, rnd_w:rnd_w + self.HQ_size, :]

        # augmentation - flip, rotate
        rlt = util.augment([img_HQ, img_LQ], self.opt['use_flip'], self.opt['use_rot'])
        img_HQ = rlt[0]
        img_LQ = rlt[1]

        # stack LQ images to NHWC, N is the frame number
        img_HQs = np.stack([img_HQ], axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_HQs = img_HQs[:, :, :, [2, 1, 0]]
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        img_HQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQs,
                                                                     (0, 3, 1, 2)))).float()

        return {'LQ': img_LQ, 'HQs': img_HQs}

    def __len__(self):
        return len(self.HQ_paths)
