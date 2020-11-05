# -*- coding: utf-8 -*-
import argparse
import torch
import os
import time
import h5py

from config import config
from models import get_model
from feature_extractor import FeatureExtractor2D, FeatureExtractor3D


def parse_args():
    parser = argparse.ArgumentParser(description="Video Feature Extractor")
    parser.add_argument('-v', '--video-dpath', type=str, 
                        help="The directory path of videos (frames).")
    parser.add_argument('-m', '--model', type=str, default='resnet50',
                        help="The name of model from which you extractfeatures.")
    parser.add_argument('-b', '--batch-size', type=int, default=32, 
                        help="The batch size.")
    parser.add_argument('-s', '--stride', type=int, default=5, 
                        help="Extract feature from every <s> frames.")
    parser.add_argument('-o', '--save-dir', type=str, default='./datasets/feature',
                        help="The file path of extracted feature.")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA device.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    config = config[args.model]

    if not args.no_cuda and torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    device = torch.device(args.device)

    model = get_model(args.model, pretrained=True)
    model.to(device)
    model.eval()

    models_3d = ['c3d']
    FeatureExtractor = FeatureExtractor3D if args.model in models_3d else FeatureExtractor2D
    extractor = FeatureExtractor(
        stride=args.stride,
        mean=config.mean,
        std=config.std,
        resize_to=config.resize_to,
        crop_to=config.crop_to,
        model=model,
        batch_size=args.batch_size)

    videos = os.listdir(args.video_dpath)
    h5 = h5py.File(args.save_dir, 'w') if not os.path.exists(args.save_dir) else h5py.File(args.save_dir, 'r+')
    for video in videos:
        video_name = os.path.splitext(video)[0]
        video_path = os.path.join(args.video_dpath, video)
        feats = extractor(video_path)
        if feats is not None:
            h5[video_id] = feats
    h5.close()


