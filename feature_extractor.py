# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data

from abc import ABC
from utils.video_loader import VideoDataset


class FeatureExtractor(ABC):
    def __init__(self, stride, mean, std, resize_to, crop_to, video_type, *, model, batch_size):
        self.model = model
        self.batch_size = batch_size

        self.dataset = VideoDataset(
            stride=stride,
            mean=mean,
            std=std,
            resize_to=resize_to,
            crop_to=crop_to,
            type=video_type,
        )

    def __call__(self, video_path):
        loader = data.DataLoader(
            dataset=self.dataset(video_path),
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
        )
        feats_list = []
        for i, images in enumerate(loader):
            images = images.cuda()
            with torch.no_grad():
                feat = self.model(images)
            feats_list.append(feat)
        feats = torch.cat(feats_list, dim=0)
        feats = feats.data.cpu().numpy()

        return feats


class FeatureExtractor2D(FeatureExtractor):
    def __init__(self, stride, mean, std, resize_to, crop_to, **kwargs):
        super(FeatureExtractor2D, self).__init__(stride, mean, std, resize_to, crop_to, 'frame', **kwargs)

class FeatureExtractor3D(FeatureExtractor):
    def __init__(self, stride, mean, std, resize_to, crop_to, **kwargs):
        super(FeatureExtractor3D, self).__init__(stride, mean, std, resize_to, crop_to, 'clip', **kwargs)

