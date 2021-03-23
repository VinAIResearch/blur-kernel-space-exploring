import glob
import os
import os.path as osp
import shutil


def regroup(root, mode):
    all_vid_paths = sorted(glob.glob(osp.join(root, "**", mode)))
    print("Number of videos: {}".format(len(all_vid_paths)))

    for vid_idx, vid_path in enumerate(all_vid_paths):
        print("Process video #{}/{}".format(vid_idx, len(all_vid_paths)))
        target_vid_path = osp.join("../datasets/GOPRO/train_regrouped", mode, "{:03d}".format(vid_idx))
        os.makedirs(target_vid_path, exist_ok=True)
        all_frame_paths = sorted(glob.glob(osp.join(vid_path, "*.png")))

        for frame_idx, frame_paths in enumerate(all_frame_paths):
            target_frame_path = osp.join(target_vid_path, "{:08d}.png".format(frame_idx))
            shutil.copyfile(frame_paths, target_frame_path)


if __name__ == "__main__":
    regroup("../datasets/GOPRO/train", "sharp")
    regroup("../datasets/GOPRO/train", "blur")
