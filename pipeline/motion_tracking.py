import glob
import os
import shutil
import subprocess
import sys
from collections import OrderedDict

import cv2
import torch
from tqdm import tqdm

project_dir = os.environ.get("PROJECT_DIR")
sys.path.append(project_dir)
sys.path.append(f'{project_dir}/includes/RAFT/core')

from raft import RAFT

from includes.RAFT.core.utils import flow_viz
from includes.RAFT.core.utils.utils import InputPadder

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


def get_frames(video_path, frames_dir):

    prefix_name = os.path.basename(video_path).split(".")[0]
    output_pattern = os.path.join(frames_dir, f"frame_%06d.jpg")
    ffmpeg_command = f"ffmpeg -i {video_path} -vf fps=20 {output_pattern}"
    subprocess.run(ffmpeg_command, shell=True)


def process_img(img, device):
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)


def load_model(weights_path, args):
    model = RAFT(args)
    pretrained_weights = torch.load(weights_path,
                                    map_location=torch.device("cpu"))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(pretrained_weights)
    model.to(mps_device)
    return model


def inference(model,
              frame1,
              frame2,
              device,
              pad_mode='sintel',
              iters=12,
              flow_init=None,
              upsample=True,
              test_mode=True):

    model.eval()
    with torch.no_grad():
        # preprocess
        frame1 = process_img(frame1, device)
        frame2 = process_img(frame2, device)

        padder = InputPadder(frame1.shape, mode=pad_mode)
        frame1, frame2 = padder.pad(frame1, frame2)

        # predict flow
        if test_mode:
            flow_low, flow_up = model(frame1,
                                      frame2,
                                      iters=iters,
                                      flow_init=flow_init,
                                      upsample=upsample,
                                      test_mode=test_mode)

            return flow_low, flow_up

        else:
            flow_iters = model(frame1,
                               frame2,
                               iters=iters,
                               flow_init=flow_init,
                               upsample=upsample,
                               test_mode=test_mode)

            return flow_iters


def get_viz(flo):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    return flow_viz.flow_to_image(flo)


# sketchy class to pass to RAFT
class Args():

    def __init__(self,
                 model='',
                 path='',
                 small=False,
                 mixed_precision=True,
                 alternate_corr=False):
        self.model = model
        self.path = path
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr

    """ Sketchy hack to pretend to iterate through the class objects """

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


def create_optical_flow(frames_folder, output_dir, weight_path, DEVICE):
    model = load_model(weight_path, args=Args())
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all frames in the folder
    frame_paths = sorted(glob.glob(os.path.join(frames_folder, '*.jpg')))
    frame_numbers = len(frame_paths)

    # Set up image saving directory

    for curr_idx in tqdm(range(1, frame_numbers)):
        prev_img = cv2.imread(frame_paths[curr_idx - 1])
        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)

        curr_img = cv2.imread(frame_paths[curr_idx])
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)

        flow_lo_cold, flow_up_cold = inference(model,
                                               prev_img,
                                               curr_img,
                                               device=DEVICE,
                                               pad_mode='kitti',
                                               flow_init=None,
                                               iters=20,
                                               test_mode=True)

        im_shape = curr_img.shape

        optical_flow_viz = get_viz(flow_up_cold)

        #cv2.resize(optical_flow_viz, im_shape)

        image_path = os.path.join(output_dir,
                                  f'frame_{str(curr_idx).zfill(3)}.png')

        cv2.imwrite(image_path, optical_flow_viz)

    print(f"Images saved at: {output_dir}")
