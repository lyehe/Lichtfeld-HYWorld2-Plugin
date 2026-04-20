"""Visual utilities — slimmed by the LFS plugin.

Upstream ``visual_util.py`` pulls in matplotlib, trimesh, scipy and
requests for GLB-mesh export and a self-rolled HTTP downloader used by
``gradio_app.py``. The LFS plugin only runs inference, so those features
are not needed. This file keeps just what ``inference_utils`` imports:

    segment_sky, run_skyseg, download_file_from_url

``download_file_from_url`` is never reached in the plugin flow (we
pre-load the skyseg ONNX session via ``core.downloads``), so it uses a
lazy ``urllib.request`` import and avoids adding ``requests`` as a dep.
"""
from __future__ import annotations

import copy
import os

import cv2
import numpy as np


def segment_sky(image_or_path, onnx_session):
    """Segment sky from an image with an ONNX model.

    Returns a mask where 255 = non-sky, 0 = sky.
    """
    if isinstance(image_or_path, (str, os.PathLike)):
        image = cv2.imread(str(image_or_path))
    else:
        image = image_or_path
    result_map = run_skyseg(onnx_session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255
    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """Run sky-segmentation inference (BGR input)."""
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    return onnx_result.astype("uint8")


def download_file_from_url(url, filename):
    """Download a URL to a file.

    Only reached when the upstream hyworld2 code falls back to self-
    downloading the skyseg ONNX. In the LFS plugin flow we always
    pre-load the session via ``core.downloads`` so this path is dead
    — but we keep the symbol importable and use ``urllib.request`` to
    avoid a ``requests`` dependency if something unexpected invokes it.
    """
    import urllib.request
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename} successfully.")
    except Exception as exc:
        print(f"Error downloading file: {exc}")
