# reference: https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive

import argparse
import os

import requests


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def download_file_from_server(server, subset, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    session = requests.Session()
    if server == "google":
        URL = "https://docs.google.com/uc?export=download"
        params = {"id": ids[subset]}
    elif server == "snu":
        URL = "https://cv.snu.ac.kr/~snah/Deblur/dataset/REDS/" + subset + ".zip"
        params = {}

    response = session.get(URL, params=params, stream=True)
    token = get_confirm_token(response)
    if token:
        params["confirm"] = token
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


parser = argparse.ArgumentParser(
    description="Download GoPro dataset from google drive to current folder", allow_abbrev=False
)

parser.add_argument("--server", type=str, default="snu", choices=("google", "snu"), help="download server choice.")
parser.add_argument("--all", action="store_true", help="download full GoPro dataset")
parser.add_argument("--train_sharp", action="store_true", help="download train_sharp.zip")
parser.add_argument("--train_blur", action="store_true", help="download train_blur.zip")
parser.add_argument("--train_blur_comp", action="store_true", help="download train_blur_comp.zip")
parser.add_argument("--train_sharp_bicubic", action="store_true", help="download train_sharp_bicubic.zip")
parser.add_argument("--train_blur_bicubic", action="store_true", help="download train_blur_bicubic.zip")
parser.add_argument("--val_sharp", action="store_true", help="download val_sharp.zip")
parser.add_argument("--val_blur", action="store_true", help="download val_blur.zip")
parser.add_argument("--val_blur_comp", action="store_true", help="download val_blur_comp.zip")
parser.add_argument("--val_sharp_bicubic", action="store_true", help="download val_sharp_bicubic.zip")
parser.add_argument("--val_blur_bicubic", action="store_true", help="download val_blur_bicubic.zip")
parser.add_argument("--test_blur", action="store_true", help="download test_blur.zip")
parser.add_argument("--test_blur_comp", action="store_true", help="download test_blur_comp.zip")
parser.add_argument("--test_sharp_bicubic", action="store_true", help="download test_sharp_bicubic.zip")
parser.add_argument("--test_blur_bicubic", action="store_true", help="download test_blur_bicubic.zip")

args = parser.parse_args()

ids = {"GoPro": "1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2"}

# Download files in GoPro directory
if os.path.basename(os.getcwd()) == "GoPro":
    root_dir = "."
else:
    os.makedirs("GoPro", exist_ok=True)
    root_dir = "GoPro"

for subset in ids:
    argdict = args.__dict__
    if args.all or argdict[subset]:
        filename = "{}/{}.zip".format(root_dir, subset)
        servername = "Google Drive" if args.server == "google" else "SNU CVLab"
        print("Downloading {}.zip from {}".format(subset, servername))
        download_file_from_server(args.server, subset, filename)  # download the designated subset
