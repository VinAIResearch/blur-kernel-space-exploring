import torch
import lmdb
import numpy as np
import os
import random
import options.options as options
import logging
import argparse

from models.kernel_wizard import KernelWizard
import utils.util as util
import data.util as data_util


def read_image(env, key, x, y, h, w):
    img = data_util.read_img(env, key, (3, 720, 1280))
    img = np.transpose(img[x:x+h, y:y+w, [2, 1, 0]], (2, 0, 1))
    return img


def main():
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Kernel extractor testing')

    parser.add_argument('--source_H', action='store',
                        help='source image height',
                        type=int, default=None)
    parser.add_argument('--source_W', action='store',
                        help='source image width',
                        type=int, default=None)
    parser.add_argument('--target_H', action='store',
                        help='target image height',
                        type=int, default=None)
    parser.add_argument('--target_W', action='store',
                        help='target image width',
                        type=int, default=None)

    parser.add_argument('--model_path', action='store',
                        help='model path',
                        type=str, default=None)
    parser.add_argument('--LQ_root', action='store',
                        help='low-quality dataroot',
                        type=str, default=None)
    parser.add_argument('--HQ_root', action='store',
                        help='high-quality dataroot',
                        type=str, default=None)
    parser.add_argument('--save_path', action='store',
                        help='save path',
                        type=str, default=None)
    parser.add_argument('--yml_path', action='store',
                        help='yml path',
                        type=str, default=None)

    args = parser.parse_args()

    model_path = args.model_path
    LQ_root = args.LQ_root
    HQ_root = args.HQ_root
    save_path = args.save_path
    source_H, source_W = args.source_H, args.source_W
    target_H, target_W = args.target_H, args.target_W
    yml_path = args.yml_path

    # model_path = '../experiments/GOPRO_wsharp_woVAE/models/300000_G.pth'
    # LQ_root = '../datasets/GOPRO/train_blur_linear.lmdb'
    # HQ_root = '../datasets/GOPRO/train_sharp.lmdb'
    # save_path = '../results/GOPRO_wsharp_woVAE'
    # source_H, source_W = 720, 1280
    # target_H, target_W = 256, 256

    # Initializing logger
    logger = logging.getLogger('base')
    os.makedirs(save_path, exist_ok=True)
    util.setup_logger('base', save_path, 'test', level=logging.INFO,
                      screen=True, tofile=True)
    logger.info('LQ root: {}'.format(LQ_root))
    logger.info('HQ root: {}'.format(HQ_root))
    logger.info('model path: {}'.format(model_path))

    # Initializing mode
    logger.info('Loading model...')
    opt = options.parse(yml_path)['KernelWizard']
    model = KernelWizard(opt)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    logger.info('Done')

    # processing data
    HQ_env = lmdb.open(HQ_root, readonly=True, lock=False, readahead=False,
                       meminit=False)
    LQ_env = lmdb.open(LQ_root, readonly=True, lock=False, readahead=False,
                       meminit=False)
    paths_HQ, _ = data_util.get_image_paths('lmdb', HQ_root)
    print(len(paths_HQ))

    psnr_avg = 0

    num_samples = 100

    for i in range(num_samples):
        key = np.random.choice(paths_HQ)

        rnd_h = random.randint(0, max(0, source_H - target_H))
        rnd_w = random.randint(0, max(0, source_W - target_W))

        source_LQ = read_image(LQ_env, key, rnd_h, rnd_w, target_H, target_W)
        source_HQ = read_image(HQ_env, key, rnd_h, rnd_w, target_H, target_W)

        source_LQ = torch.Tensor(source_LQ).unsqueeze(0).to(device)
        source_HQ = torch.Tensor(source_HQ).unsqueeze(0).to(device)

        with torch.no_grad():
            kernel_mean, kernel_sigma = model(source_HQ, source_LQ)
            kernel = kernel_mean + kernel_sigma * torch.randn_like(kernel_mean)
            fake_LQ = model.adaptKernel(source_HQ, kernel)

        LQ_img = util.tensor2img(source_LQ)
        fake_LQ_img = util.tensor2img(fake_LQ)

        psnr = util.calculate_psnr(LQ_img, fake_LQ_img)

        print('PSNR: {:.2f}db'.format(psnr))
        psnr_avg += psnr

    print('Average PSNR: {:.2f}db'.format(psnr_avg / num_samples))


main()
