# awesome-video-feature-extractor
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

Video feature extracor for **your own datasets**.

## Environment
- Ubuntu 16.04
- CUDA 9.0
- cuDNN 7.5.1
- Python 3.x

## Usage

### Video DownLoad (optional)

Download videos from urls and put them in to a directory, such as:

```python
python3 ./tools/video_download.py ./datasets/video_urls \
    --video-dir ./datasets/video \
    --video-type 'mp4' \
    --threads 8 \
```

### Video CutFrame (optional)

Extract frames from video. Such as extract 1 frames per second and using 8 threads:

```python
python3 ./tools/video2frame.py ./datasets/video_list \
    --frame-dir ./datasets/frame \
    --fps 1 \
    --threads 8
```

### Extract Features

Extract features from videos, such as:

```
python3 main.py \
    --video-dpath ./datasets/video \
    --model resnet50 \
    --batch-size 32 \
    --save-dir ./datasets/feature
```

## References

- [pytorch-video-feature-extractor](https://github.com/hobincar/pytorch-video-feature-extractor)
- [video_features](https://github.com/v-iashin/video_features)
- [video2frame](https://github.com/jinyu121/video2frame)

<!--
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]
-->

[python-image]: https://img.shields.io/badge/Python-2.x|3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.1-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: https://img.shields.io/badge/Apache-2.0-blue.svg
[lic-url]: https://github.com/Tramac/awesome-video-feature-extractor/blob/master/LICENSE
