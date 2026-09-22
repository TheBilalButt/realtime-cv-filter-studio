"""
Real Time Computer Vision Filters Implementation.
All filters operate on 8-bit RGB NumPy arrays and utilize high-performance
vectorized OpenCV and NumPy functions without slow Python loops.

Built by Bilal Butt
"""

import cv2
import numpy as np
from typing import Literal


def ensure_odd(value: int, min_val: int = 3) -> int:
    """Ensures an integer is odd and at least min_val (required for blur kernels)."""
    val = max(min_val, int(value))
    return val if val % 2 != 0 else val + 1


def apply_original(image: np.ndarray) -> np.ndarray:
    """Returns an identical copy of the input RGB image."""
    return image.copy()


def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts the RGB image to grayscale, returned in 3-channel RGB for display consistency."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 15, sigma: float = 0.0) -> np.ndarray:
    """Applies Gaussian Blur with an odd kernel size."""
    k = ensure_odd(kernel_size)
    return cv2.GaussianBlur(image, (k, k), sigmaX=float(sigma))


def apply_median_blur(image: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """Applies Median Blur with an odd kernel size (effective for salt-and-pepper noise)."""
    k = ensure_odd(kernel_size)
    return cv2.medianBlur(image, k)


def apply_sharpen(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """
    Applies an unsharp masking convolution kernel parameterized by strength.
    Uses an unsharp mask: result = image * (1 + strength) - blur * strength
    for smooth, artifact-free sharpening.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3.0)
    sharpened = cv2.addWeighted(image, 1.0 + float(strength), blurred, -float(strength), 0)
    return cv2.convertScaleAbs(sharpened)


def apply_edge_detection(
    image: np.ndarray,
    method: Literal["Sobel", "Laplacian", "Prewitt"] = "Sobel",
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Detects edges using Sobel, Laplacian, or Prewitt operators.
    Returns a 3-channel RGB edge representation.
    """
    k = ensure_odd(kernel_size, min_val=1)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if method == "Laplacian":
        k_lap = ensure_odd(kernel_size, min_val=1)
        edges = cv2.Laplacian(gray, cv2.CV_16S, ksize=k_lap)
        edges = cv2.convertScaleAbs(edges)
    elif method == "Prewitt":
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        px = cv2.filter2D(gray, -1, kernelx)
        py = cv2.filter2D(gray, -1, kernely)
        edges = cv2.addWeighted(px, 0.5, py, 0.5, 0)
    else:  # Sobel (default)
        k_sobel = 3 if k > 7 else k
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=k_sobel)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=k_sobel)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def apply_canny(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Applies the Canny edge detector with tunable low and high hysteresis thresholds.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, int(low_threshold), int(high_threshold))
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def apply_threshold(
    image: np.ndarray,
    thresh: int = 127,
    max_val: int = 255,
    threshold_type: Literal["Binary", "Binary Inverted", "Otsu"] = "Binary",
) -> np.ndarray:
    """
    Applies standard thresholding (Binary, Inverted, or Automatic Otsu).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if threshold_type == "Binary Inverted":
        _, binary = cv2.threshold(gray, thresh, max_val, cv2.THRESH_BINARY_INV)
    elif threshold_type == "Otsu":
        _, binary = cv2.threshold(gray, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, thresh, max_val, cv2.THRESH_BINARY)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def apply_adaptive_threshold(
    image: np.ndarray,
    max_val: int = 255,
    block_size: int = 11,
    c_val: int = 2,
    method: Literal["Gaussian", "Mean"] = "Gaussian",
) -> np.ndarray:
    """
    Applies adaptive thresholding for varying illumination conditions.
    """
    b_size = ensure_odd(block_size, min_val=3)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "Gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    adaptive = cv2.adaptiveThreshold(
        gray, max_val, adaptive_method, cv2.THRESH_BINARY, b_size, c_val
    )
    return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)


def apply_brightness(image: np.ndarray, brightness: int = 30) -> np.ndarray:
    """Adjusts image brightness (-100 to +100)."""
    # cv2.convertScaleAbs efficiently computes: saturate(image * alpha + beta)
    return cv2.convertScaleAbs(image, alpha=1.0, beta=int(brightness))


def apply_contrast(image: np.ndarray, contrast: float = 1.5) -> np.ndarray:
    """Adjusts image contrast multiplier (0.1 to 3.0)."""
    return cv2.convertScaleAbs(image, alpha=float(contrast), beta=0)


def apply_saturation(image: np.ndarray, saturation: float = 1.5) -> np.ndarray:
    """
    Adjusts color saturation using the HSV color space.
    saturation < 1.0 reduces saturation (grayscale at 0.0), > 1.0 boosts colors.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = hsv[..., 1] * float(saturation)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def apply_negative(image: np.ndarray) -> np.ndarray:
    """Inverts all pixel color intensities (digital negative)."""
    return cv2.bitwise_not(image)


def apply_sepia(image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    Applies an authentic warm sepia photographic tone matrix.
    intensity blends smoothly between original (0.0) and full sepia (1.0).
    """
    sepia_matrix = np.array(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ],
        dtype=np.float32,
    )
    # cv2.transform performs matrix multiplication across channels
    sepia = cv2.transform(image, sepia_matrix)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    if intensity >= 0.99:
        return sepia
    elif intensity <= 0.01:
        return image.copy()

    return cv2.addWeighted(sepia, float(intensity), image, 1.0 - float(intensity), 0)


def apply_emboss(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Creates a 3D embossed relief effect with highlight and shadow edges.
    """
    kernel = np.array(
        [
            [-2, -1, 0],
            [-1,  1, 1],
            [ 0,  1, 2],
        ],
        dtype=np.float32,
    ) * float(strength)

    # 128 gray baseline bias
    embossed = cv2.filter2D(image, -1, kernel, delta=128)
    return cv2.convertScaleAbs(embossed)


def apply_cartoon(
    image: np.ndarray,
    num_bilateral: int = 3,
    num_colors: int = 8,
    edge_kernel: int = 7,
) -> np.ndarray:
    """
    Produces a high-quality cartoon / anime cel-shaded look:
    1. Edge detection via adaptive thresholding
    2. Color quantization / flattening via iterative bilateral filtering
    3. Merging quantized colors with bold edge lines
    """
    k = ensure_odd(edge_kernel, min_val=3)

    # Step 1: Detect clean edges
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, k, 7
    )
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Step 2: Smooth textures while keeping major edges sharp using bilateral filter
    color = image.copy()
    for _ in range(num_bilateral):
        color = cv2.bilateralFilter(color, d=7, sigmaColor=75, sigmaSpace=75)

    # Step 3: Fast color quantization (palette reduction)
    if num_colors > 0 and num_colors < 256:
        factor = 256 // num_colors
        color = (color // factor) * factor + (factor // 2)

    # Step 4: Combine flattened colors with sharp edges
    cartoon = cv2.bitwise_and(color, edges_rgb)
    return cartoon


def apply_document_scanner(
    image: np.ndarray,
    contrast: float = 1.25,
    brightness: int = 15,
    mode: Literal["Color Document", "Black and White Scan"] = "Color Document",
) -> np.ndarray:
    """
    Cleans written pages, whiteboards, notes, and documents in real time:
    - Removes shadows, yellowish tints, creases, and uneven ambient lighting
    - Makes the paper background crystal-clear pure white (255, 255, 255)
    - Keeps ink (handwriting, pen strokes, print) sharp and legible
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Estimate background illumination surface
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)

    if mode == "Black and White Scan":
        diff = 255 - cv2.absdiff(gray, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        binary = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    # Color Document mode: normalize RGB channels by estimated luminance background
    bg_f = np.maximum(bg.astype(np.float32), 1.0)
    norm = (image.astype(np.float32) / bg_f[:, :, None]) * 255.0
    stretched = np.clip(norm * float(contrast) + int(brightness), 0, 255).astype(np.uint8)
    return stretched

