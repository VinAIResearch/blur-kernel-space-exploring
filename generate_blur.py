import argparse
import torch
import cv2
import numpy as np

import options.options as options
import utils.util as util
from models.kernel_encoding.kernel_wizard import KernelWizard


def main():
    device = torch.device("cuda")

    parser = argparse.ArgumentParser(description="Kernel extractor testing")

    parser.add_argument("--image_path", action="store", help="image path", type=str, required=True)
    parser.add_argument("--model_path", action="store", help="model path", type=str, required=True)
    parser.add_argument("--yml_path", action="store", help="yml path", type=str, required=True)

    args = parser.parse_args()

    model_path = args.model_path
    image_path = args.image_path
    yml_path = args.yml_path

    # Initializing mode
    opt = options.parse(yml_path)["KernelWizard"]
    model = KernelWizard(opt)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    HQ = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.
    HQ = np.transpose(HQ, (2, 0, 1))
    HQ_tensor = torch.Tensor(HQ).unsqueeze(0).to(device).cuda()

    with torch.no_grad():
        kernel = torch.randn((1, 512, 2, 2)).cuda()
        LQ_tensor = model.adaptKernel(HQ_tensor, kernel)

    LQ_img = util.tensor2img(LQ_tensor)

    cv2.imwrite('blur.png', LQ_img)


main()
