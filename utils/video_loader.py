import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from .commons import sample_frames_metafunc, sample_clips_metafunc, preprocess_frame_metafunc, preprocess_clip_metafunc


class VideoDataset(object):
    def __init__(self, stride, mean, std, resize_to, crop_to, type='frame'):
        self.sample_func = sample_frames_metafunc(stride)
        self.preprocess_func = preprocess_frame_metafunc(mean, std, resize_to, crop_to)
        """
        self.preprocess_func2 = transforms.Compose([
            transforms.Resize(resize_to),
            #transforms.CenterCrop(crop_to),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        """

        if type == 'clip':
            self.sample_func = sample_clips_metafunc(stride)
            self.preprocess_func = preprocess_clip_metafunc(mean, std, resize_to, crop_to)

    def __call__(self, video_fpath):
        X = self.sample_func(video_fpath)
        if X.shape[0] == 0:
            return None

        self.samples = X

        return self

    def __getitem__(self, idx):
        sample = self.samples[idx]
        #sample = Image.fromarray(sample, mode='RGB')
        sample = self.preprocess_func(sample)

        if len(sample.shape) == 3: # HxWxC --> CxHxW
            sample = sample.transpose((2, 0, 1))
        elif len(sample.shape) == 4: # NxHxWxC --> CxNxHxW
            sample = sample.transpose((3, 0, 1, 2))
        else:
            raise NotImplementedError("Unknown sample of shape {}".format(sample.shape))

        return torch.from_numpy(sample).float()

    def __len__(self):
        return self.samples.shape[0]

