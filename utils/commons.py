import os
import cv2
import numpy as np
from typing import Any, List, Sequence
from PIL import Image


def resize_frame(image, size, interpolation=cv2.INTER_LINEAR):
    # size (h, w)
    image = np.asarray(image, dtype=np.float32)

    if not (isinstance(size, int) or (isinstance(size, Sequence) and len(size) in (1, 2))):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if isinstance(size, int) or len(size) == 1:
        if isinstance(size, Sequence):
            size = size[0]
        if (width <= height and width == size) or (height <= width and height == size):
            return image
        if width < height:
            ow = size
            oh = int(size * height / width)
            return cv2.resize(image, (oh, ow), interpolation=interpolation)
        else:
            oh = size
            ow = int(size * width / height)
            return cv2.resize(image, (oh, ow), interpolation=interpolation)
    else:
        return cv2.resize(image, size, interpolation=interpolation)


def center_crop_frame(image, th, tw):
    if len(image.shape) == 3:
        h, w, c = image.shape
    elif len(image.shape) == 4:
        _, h, w, c = image.shape
    else:
        NotImplementedError("Unknown image of shape {}".format(image.shape))

    x1 = int(round((h - th) / 2.))
    y1 = int(round((w - tw) / 2.))
    if len(image.shape) == 3:
        crop_image = image[x1:x1 + th, y1:y1 + tw, :]
    else:
        crop_image = image[:, x1:x1 + th, y1:y1 + tw, :]
    return crop_image


def sample_frames_metafunc(stride):
    def sample_frames(video_path):
        frames = []
        # please match video2frame
        frame_dir = os.path.splitext(video_path)[0].replace('video', 'frame', 1)
        if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) > 0:
            for frame_name in sorted(os.listdir(frame_dir)):
                frame_path = os.path.join(frame_dir, frame_name)
                # plz note the diff between Image and cv2
                # keep the same method in the train and infer.
                frame = np.array(Image.open(frame_path).convert('RGB'))
                frames.append(frame)
            frames = np.array(frames)
        else:
            try:
                cap = cv2.VideoCapture(video_path)
            except:
                print('Can not open %s.' % video_path)
                pass
    
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if ret is False:
                    break
                frame = frame[:, :, ::-1] # BGR --> RGB
                frames.append(frame)
                frame_count += 1
    
            # remove the start and end 8 frames
            indices = list(range(8, frame_count - 7, stride))
    
            frames = np.array(frames)
            frames = frames[indices]
        return frames

    return sample_frames


def sample_clips_metafunc(stride):
    def sample_clips(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
        except:
            print('Can not open %s.' % video_path)
            pass
    
        frames = []
        frame_count = 0
    
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            frame = frame[:, :, ::-1] # BGR --> RGB
            frames.append(frame)
            frame_count += 1
    
        indices = list(range(8, frame_count - 7, stride))

        frames = np.array(frames)
        clip_list = []
        for index in indices:
            clip_list.append(frames[index - 8: index + 8])
        clip_list = np.array(clip_list)
        return clip_list, frame_count

    return sample_clips


def preprocess_frame_metafunc(mean, std, resize_to, crop_to):
    def preprocess_frame(image):
        image = resize_frame(image, resize_to)
        image /= 255.
        image -= np.asarray(mean)
        image /= np.asarray(std)
        if crop_to is not None:
            image = center_crop_frame(image, *crop_to)
        return image
    
    return preprocess_frame


def preprocess_clip_metafunc(mean, std, resize_to, crop_to):
    if crop_to is not None:
        mean = center_crop_frame(mean, *crop_to)

    def preprocess_clip(clip):
        clip = np.array([resize_frame(frame, resize_to) for frame in clip])
        if crop_to is not None:
            clip = center_crop_frame(clip, *crop_to)
        clip /= 255.
        clip -= mean
        clip /= np.asarray(std)
        return clip

    return preprocess_clip

