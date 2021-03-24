import argparse
import logging
import os
import torch
import cv2
import glob

import options.options as options
import utils.util as util
from models.kernel_wizard import KernelWizard


def main():
    device = torch.device("cuda")

    parser = argparse.ArgumentParser(description="Kernel extractor testing")

    parser.add_argument("--model_path", action="store", help="model path", type=str, default=None)
    parser.add_argument("--LQ1_root", action="store", help="first low-quality dataroot", type=str, default=None)
    parser.add_argument("--HQ1_root", action="store", help="first high-quality dataroot", type=str, default=None)
    parser.add_argument("--LQ2_root", action="store", help="second low-quality dataroot", type=str, default=None)
    parser.add_argument("--HQ2_root", action="store", help="second high-quality dataroot", type=str, default=None)
    parser.add_argument("--yml_path", action="store", help="yml path", type=str, default=None)

    args = parser.parse_args()

    # Initializing logger
    logger = logging.getLogger("base")
    os.makedirs(args.save_path, exist_ok=True)
    util.setup_logger("base", args.save_path, "test", level=logging.INFO, screen=True, tofile=True)
    logger.info("first LQ root: {}".format(args.LQ1_root))
    logger.info("first HQ root: {}".format(args.HQ1_root))
    logger.info("second LQ root: {}".format(args.LQ2_root))
    logger.info("second HQ root: {}".format(args.HQ2_root))
    logger.info("model path: {}".format(args.model_path))

    # Initializing model
    logger.info("Loading model...")
    opt = options.parse(args.yml_path)["KernelWizard"]
    model = KernelWizard(opt)
    model.eval()
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    logger.info("Done")

    # processing data
    LQ1_paths = sorted(glob.glob(args.LQ1_root))
    HQ1_paths = sorted(glob.glob(args.HQ1_root))
    LQ2_paths = sorted(glob.glob(args.LQ2_root))
    HQ2_paths = sorted(glob.glob(args.HQ2_root))
    num_images = len(LQ1_paths)

    psnr_avg = 0

    for i, (LQ1_path, HQ1_path, LQ2_path, HQ2_path) in enumerate(zip(LQ1_paths, HQ1_paths, LQ2_paths, HQ2_paths)):
        LQ1 = cv2.cvtColor(cv2.imread(LQ1_path), cv2.COLOR_BGR2RGB)
        HQ1 = cv2.cvtColor(cv2.imread(HQ1_path), cv2.COLOR_BGR2RGB)
        LQ2 = cv2.cvtColor(cv2.imread(LQ2_path), cv2.COLOR_BGR2RGB)
        HQ2 = cv2.cvtColor(cv2.imread(HQ2_path), cv2.COLOR_BGR2RGB)

        LQ1_tensor = torch.Tensor(LQ1).unsqueeze(0).to(device)
        HQ1_tensor = torch.Tensor(HQ1).unsqueeze(0).to(device)
        HQ2_tensor = torch.Tensor(HQ2).unsqueeze(0).to(device)

        with torch.no_grad():
            kernel_mean, kernel_sigma = model(HQ1_tensor, LQ1_tensor)
            kernel = kernel_mean + kernel_sigma * torch.randn_like(kernel_mean)
            fake_LQ2 = model.adaptKernel(HQ2_tensor, kernel)

        psnr = util.calculate_psnr(LQ2, fake_LQ2)

        logger.info("Transferring PSNR of image set #{}/{}: {:.2f}db".format(i, num_images, psnr))
        psnr_avg += psnr

    logger.info("Average transferring PSNR: {:.2f}db".format(psnr_avg / num_images))


main()
