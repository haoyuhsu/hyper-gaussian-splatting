import argparse
import json
import os
import numpy as np
from tqdm import tqdm

# Transform "transforms.json" in each scene to NeRFStudio format.

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='./data_objaverse/')
    args = parser.parse_args()

    dir_list = sorted([attr for attr in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, attr))])

    for scene in tqdm(dir_list):
        
        root_dir = os.path.join(args.input_dir, scene)

        images_dir = os.path.join(root_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Copy images.
        os.system("rm -r '{}'".format(images_dir))
        os.system("cp -r '{}' '{}'".format(os.path.join(root_dir, "train/"), images_dir))

        transforms_path = os.path.join(root_dir, "transforms.json")
        old_transforms_path = os.path.join(root_dir, "transforms_old.json")

        # Skip if "transforms.json" does not exist.
        if not os.path.exists(old_transforms_path):
            # rename "transforms.json" to "transforms_old.json"
            os.rename(transforms_path, old_transforms_path)

        with open(old_transforms_path, "r") as f:
            transforms = json.load(f)

        new_transforms = {}

        # store intrinsics
        h, w = 800, 800
        camera_angle_x = transforms["camera_angle_x"]
        focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

        new_transforms["camera_model"] = "OPENCV"
        new_transforms["fl_x"] = focal
        new_transforms["fl_y"] = focal
        new_transforms["cx"] = w / 2.0
        new_transforms["cy"] = h / 2.0
        new_transforms["w"] = w
        new_transforms["h"] = h

        # store extrinsics
        frames_info = transforms["frames"]
        new_frames_info = []
        for frame_info in frames_info:
            frame_id = frame_info["file_path"].split("/")[-1]
            transform_matrix = frame_info["transform_matrix"]
            
            new_frame_info = {}
            new_frame_info["file_path"] = os.path.join("images", frame_id + ".png")
            new_frame_info["transform_matrix"] = transform_matrix
            # transform_matrix = np.array(transform_matrix) @ np.diag([1, -1, -1, 1])  # OpenGL to OpenCV camera
            # new_frame_info["transform_matrix"] = transform_matrix.tolist()  # C2W (4, 4) OpenCV
            new_frames_info.append(new_frame_info)
        new_transforms["frames"] = new_frames_info

        # store final transforms.json
        with open(os.path.join(root_dir, "transforms.json"), "w") as f:
            json.dump(new_transforms, f, indent=4)