import argparse
import glob
import os
import shutil
import sys
import time

import torch

from pipeline.motion_tracking import create_optical_flow, get_frames
from program_utils import config

project_dir = os.environ.get("PROJECT_DIR")
sys.path.append(project_dir)

# set up device
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

DEVICE = mps_device


def get_frames_from_video(video_dir, frames_dir):
    video_paths = glob.glob(f"{video_dir}/*.mp4")
    count = 0
    number_of_videos = len(video_paths)

    for video_path in video_paths:
        start = time.time()
        video_basename = os.path.basename(video_path).split(".")[0]
        video_frame_dir = os.path.join(frames_dir, video_basename)

        print(f"---processing video {video_basename}")

        video_frame_dir = os.path.join(frames_dir, f'{video_basename}')
        if os.path.exists(video_frame_dir):
            shutil.rmtree(video_frame_dir)

        os.makedirs(video_frame_dir, exist_ok=True)

        get_frames(video_path, video_frame_dir)

        duration = (time.time() - start) / 60
        count += 1

        print(f"-- Video {number_of_videos}/{count}complete after {duration}")


def get_optical_flow_with_raft(video_dir, raft_weight_path, frames_dir,
                               optical_flow_dir):

    video_paths = glob.glob(f"{video_dir}/*.mp4")
    count = 0
    number_of_videos = len(video_paths)

    for video_path in video_paths:
        start = time.time()

        video_basename = os.path.basename(video_path).split(".")[0]
        video_optical_flow_dir = os.path.join(optical_flow_dir, video_basename)
        video_frame_dir = os.path.join(frames_dir, video_basename)

        print(f"---processing video {video_basename}")

        if os.path.exists(video_optical_flow_dir):
            shutil.rmtree(video_optical_flow_dir)
        os.makedirs(video_optical_flow_dir, exist_ok=True)

        video_frame_dir = os.path.join(frames_dir, f'{video_basename}')

        create_optical_flow(video_frame_dir, video_optical_flow_dir,
                            raft_weight_path, DEVICE)

        duration = (time.time() - start) / 60
        count += 1
        print(f"-- Video {number_of_videos}/{count}complete after {duration}")


def get_rcnn_inference():
    return


def get_core_localization():
    return


def main(config, project_dir, frame=False, raft=False, rcnn=False, core=False):
    root = project_dir
    path_cfg = config.get("path")

    video_path = os.path.join(root, path_cfg.get("videos"))
    frames_path = os.path.join(root, path_cfg.get("frames"))
    optical_flow_images_path = os.path.join(root, path_cfg.get("raft_images"))
    raft_weight_path = os.path.join(root, path_cfg.get("raft_weight_path"))

    if frame:
        get_frames_from_video(video_path, frames_path)

    if raft:
        get_optical_flow_with_raft(video_path, raft_weight_path, frames_path,
                                   optical_flow_images_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame")
    parser.add_argument("--raft")
    parser.add_argument("--rcnn")
    parser.add_argument("--core")

    main(config, project_dir, frame=False, raft=True, rcnn=False, core=False)
