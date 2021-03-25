import argparse
import cv2
import yaml

from models.JointDeblur import JointDeblur


def main():
    parser = argparse.ArgumentParser(description="Kernel extractor testing")

    parser.add_argument("--image_path", action="store", help="image path", type=str, required=True)
    parser.add_argument("--yml_path", action="store", help="yml path", type=str, required=True)

    args = parser.parse_args()

    # Initializing mode
    with open(args.yml_path, "rb") as f:
        opt = yaml.safe_load(f)
    model = JointDeblur(opt)

    blur_img = cv2.cvtColor(cv2.imread(args.image_path))
    sharp_img = model.deblur(blur_img)

    cv2.imwrite("res.png", sharp_img)


main()
