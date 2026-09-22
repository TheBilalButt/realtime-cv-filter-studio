"""
Utility helper functions for Real Time Computer Vision Filter Studio.
Handles image conversions, resizing, benchmarking, and export.

Built by Bilal Butt
"""

import io
import time
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union


def load_image(source: Union[str, bytes, bytearray, io.BytesIO, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Loads an image from various sources and returns an RGB numpy array (uint8).
    Handles file paths, uploaded file bytes, PIL images, and existing numpy arrays.
    """
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        elif source.shape[2] == 4:
            return cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        return source.copy()

    if isinstance(source, Image.Image):
        source = source.convert("RGB")
        return np.array(source)

    if isinstance(source, (bytes, bytearray)):
        file_bytes = np.frombuffer(source, np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image from byte buffer.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if hasattr(source, "read"):
        file_bytes = np.frombuffer(source.read(), np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image from buffer stream.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if isinstance(source, str):
        bgr = cv2.imread(source)
        if bgr is None:
            raise FileNotFoundError(f"Could not load image from path: {source}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise TypeError(f"Unsupported image input type: {type(source)}")


def resize_image_max_dim(image: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    """
    Resizes an image proportionally if its maximum dimension exceeds max_dim.
    Preserves exact aspect ratio and avoids unnecessary copies if already within bounds.
    """
    if max_dim <= 0:
        return image

    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return image

    scale = max_dim / float(largest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Use INTER_AREA for downscaling (sharpest and artifact-free)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def convert_to_download_bytes(image: np.ndarray, format: str = "JPEG", quality: int = 95) -> bytes:
    """
    Encodes an RGB or RGBA image into an in-memory byte buffer for downloading.
    Supported formats: "JPEG", "PNG", "WEBP".
    """
    fmt = format.upper()
    if fmt in ["JPG", "JPEG"]:
        if image.ndim == 3 and image.shape[2] == 4:
            # Drop alpha for standard JPEG
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
        success, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    elif fmt == "PNG":
        if image.ndim == 3 and image.shape[2] == 4:
            bgra = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            success, encoded = cv2.imencode(".png", bgra, [int(cv2.IMWRITE_PNG_COMPRESSION), 4])
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
            success, encoded = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 4])
    elif fmt == "WEBP":
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if (image.ndim == 3 and image.shape[2] == 3) else image
        success, encoded = cv2.imencode(".webp", bgr, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
    else:
        raise ValueError(f"Unsupported format: {format}")

    if not success:
        raise RuntimeError("Failed to encode image to buffer.")

    return encoded.tobytes()


class PerformanceTracker:
    """
    Tracks inference / processing durations and computes smoothed FPS and latency metrics.
    """

    def __init__(self, smoothing_window: int = 10):
        self.smoothing_window = smoothing_window
        self.durations = []
        self.last_timestamp = time.perf_counter()

    def record_duration(self, duration_sec: float):
        self.durations.append(duration_sec)
        if len(self.durations) > self.smoothing_window:
            self.durations.pop(0)

    @property
    def avg_duration_ms(self) -> float:
        if not self.durations:
            return 0.0
        return (sum(self.durations) / len(self.durations)) * 1000.0

    @property
    def current_fps(self) -> float:
        if not self.durations:
            return 0.0
        avg = sum(self.durations) / len(self.durations)
        return (1.0 / avg) if avg > 0 else 0.0
