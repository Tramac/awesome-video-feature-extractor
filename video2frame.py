# -*- coding: utf-8 -*-
import sys
import argparse
import multiprocessing
import subprocess
import json
import re
import os
import concurrent

from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

ffprobe_path = "../../ffmpeg/ffprobe"
ffmpeg_path = "../../ffmpeg/ffmpeg"
ffmpeg_duration_template = re.compile(r"time=\s*(\d+):(\d+):(\d+)\.(\d+)")

def parse_args():
    parser = argparse.ArgumentParser(description='Cutframe from Video File.')
    # folders
    parser.add_argument("video_file", type=str, help="The video file")
    parser.add_argument("--frame-dir", type=str, default="./datasets/frame", help="Path to frames")
    # frame rate
    parser.add_argument("--fps", type=float, default=1, help="Sample the video at X fps")
    # threads
    parser.add_argument("--threads", type=int, default=0, help="Number of threads")

    args = parser.parse_args()
    args = EasyDict(args.__dict__)

    if args.threads and args.threads < 0:
        args.threads = max(multiprocessing.cpu_count() // 2, 1)

    return args

def read_videos(video_file):
    video_list = []
    with open(video_file) as lines:
        for line in lines:
            line = line.strip().split('\t')
            video_list.append(line[0])

    return video_list

def get_video_duration(video_file):
    cmd = [
        ffmpeg_path,
        "-i", str(video_file),
        "-f", "null", "-"
    ]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    result_all = ffmpeg_duration_template.findall(output.decode())
    if result_all:
        result = result_all[-1]
        duration = float(result[0]) * 60 * 60 \
                   + float(result[1]) * 60 \
                   + float(result[2]) \
                   + float(result[3]) * (10 ** -len(result[3]))
    else:
        duration = -1
    return duration


def get_video_meta(video_file):
    try:
        cmd = [
            ffprobe_path,
            "-show_streams",
            "-print_format", "json",
            "-v", "quiet",
            str(video_file),
        ]
        output = subprocess.check_output(cmd)
        output = json.loads(output)

        streamsbytype = {}
        for stream in output["streams"]:
            streamsbytype[stream["codec_type"].lower()] = stream

        return streamsbytype
    except:
        return {}

def video_to_frames(args, video_file, video_meta, error_when_empty=True):
    video_name = os.path.basename(video_file).split('.')[0]
    frame_dir = Path(args.frame_dir) / "{}".format(video_name)
    frame_dir.mkdir(exist_ok=True)
    try:
        video_duration = float(video_meta["video"]["duration"])
    except:
        video_duration = get_video_duration(video_file)
    cmd = [
        ffmpeg_path,
        "-loglevel", "panic",
        "-vsync", "vfr",
        "-i", str(video_file),
        "-vf", "fps={}".format(args.fps),
        "-q:v", "2",
        "{}/%5d.jpg".format(frame_dir),
    ]
    subprocess.call(cmd)

    frames = [(int(f.name.split('.')[0]), f) for f in frame_dir.iterdir()]
    frames.sort(key=lambda x: x[0])
    if error_when_empty and not frames:
        raise RuntimeError("Extract frame failed")

    return frames


def process(args, video_path):
    video_file = Path(video_path)

    if not video_file.exists():
        raise RuntimeError("Video not exists")

    video_meta = get_video_meta(video_file)
    if not video_meta:
        raise RuntimeError("Can not get video info")

    frames = video_to_frames(args, video_file, video_meta)
    return "OK"


if __name__ == '__main__':
    args = parse_args()
    Path(args.frame_dir).mkdir(exist_ok=True)

    video_list = read_videos(args.video_file)
    total = len(video_list)
    fails = []

    if args.threads > 0:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            jobs = {
                executor.submit(process, *(args, video_path)): video_path \
                    for idx, video_path in enumerate(video_list)
            }
            for future in tqdm(concurrent.futures.as_completed(jobs), total=total):
                try:
                    video_status = future.result()
                except Exception as e:
                    tqdm.write("{} : {}".format(jobs[future], e))
                    fails.append(jobs[future])
                else:
                    tqdm.write("{} : {}".format(jobs[future], video_status))
    else:
        for video_path in tqdm(video_list):
            try:
                video_status = process(args, video_path)
            except Exception as e:
                tqdm.write("{} : {}".format(video_path, e))
                fails.append(video_path)
            else:
                tqdm.write("{} : {}".format(video_path, video_status))

    print("Processed {} videos".format(total))
    if not fails:
        print("All success! Congratulations!")
    else:
        print("{} Success, {} Error".format(total - len(fails), len(fails)))
    print("All Done!")

