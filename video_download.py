# -*- coding: utf-8 -*-
import sys
import argparse
import multiprocessing
import subprocess
import json
import re
import os
import concurrent
import tempfile
import requests

from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def parse_args():
    parser = argparse.ArgumentParser(description='Video Downloader.')
    parser.add_argument("url_file", type=str, help="The video file")
    parser.add_argument("--video-dir", type=str, default="./datasets/video", help="Path to videos")
    parser.add_argument("--video-type", type=str, default="mp4", help="Video type")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--timeout", type=int, default=900, help="Number of timeout")

    args = parser.parse_args()
    args = EasyDict(args.__dict__)

    if args.threads and args.threads < 0:
        args.threads = max(multiprocessing.cpu_count() // 2, 1)

    return args

def read_videos(video_urls):
    urls = []
    with open(video_urls) as lines:
        for line in lines:
            line = line.strip().split('\t')
            urls.append(line[1])

    return urls

def process(args, url):
    url = url.replace('https://', 'http://') if url.startswith('https://') else url
    mp4_file = tempfile.NamedTemporaryFile(prefix="video_", suffix="." + args.video_type, 
                                           dir=args.video_dir, delete=False)
    try:
        response = requests.get(url, timeout=args.timeout)
    except:
        return "failed"
    if response.status_code != requests.codes.ok:
        return "failed"
    mp4_file.write(response.content)
    mp4_file.close()

    return mp4_file.name


if __name__ == '__main__':
    args = parse_args()
    Path(args.video_dir).mkdir(exist_ok=True)

    video_urls = read_videos(args.url_file)
    total = len(video_urls)
    fails = []

    if args.threads > 0:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            jobs = {
                executor.submit(process, *(args, video_url)): video_url \
                    for idx, video_url in enumerate(video_urls)
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
        for video_url in tqdm(video_urls):
            try:
                video_status = process(args, video_url)
            except Exception as e:
                tqdm.write("{} : {}".format(video_url, e))
                fails.append(video_url)
            else:
                tqdm.write("{} : {}".format(video_url, video_status))

    print("Processed {} videos".format(total))
    if not fails:
        print("All success! Congratulations!")
    else:
        print("{} Success, {} Error".format(total - len(fails), len(fails)))
    print("All Done!")

