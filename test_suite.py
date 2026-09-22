"""
Automated unit test and benchmark suite for Real Time CV Filter Studio.

Built by Bilal Butt
"""

import time
import unittest
import numpy as np
import cv2

import filters
import background
import utils


class TestFilters(unittest.TestCase):
    def setUp(self):
        # Create a standard 480x640 RGB image
        np.random.seed(42)
        self.image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def test_original(self):
        res = filters.apply_original(self.image)
        self.assertEqual(res.shape, self.image.shape)
        self.assertTrue(np.array_equal(res, self.image))

    def test_grayscale(self):
        res = filters.apply_grayscale(self.image)
        self.assertEqual(res.shape, self.image.shape)
        # All 3 channels should be equal in gray representation
        self.assertTrue(np.array_equal(res[:, :, 0], res[:, :, 1]))

    def test_gaussian_blur(self):
        res = filters.apply_gaussian_blur(self.image, kernel_size=15, sigma=1.5)
        self.assertEqual(res.shape, self.image.shape)

    def test_median_blur(self):
        res = filters.apply_median_blur(self.image, kernel_size=11)
        self.assertEqual(res.shape, self.image.shape)

    def test_sharpen(self):
        res = filters.apply_sharpen(self.image, strength=2.0)
        self.assertEqual(res.shape, self.image.shape)

    def test_edge_detection_methods(self):
        for method in ["Sobel", "Laplacian", "Prewitt"]:
            res = filters.apply_edge_detection(self.image, method=method, kernel_size=3)
            self.assertEqual(res.shape, self.image.shape)

    def test_canny(self):
        res = filters.apply_canny(self.image, low_threshold=50, high_threshold=150)
        self.assertEqual(res.shape, self.image.shape)

    def test_threshold_types(self):
        for t_type in ["Binary", "Binary Inverted", "Otsu"]:
            res = filters.apply_threshold(self.image, thresh=128, max_val=255, threshold_type=t_type)
            self.assertEqual(res.shape, self.image.shape)

    def test_adaptive_threshold(self):
        for method in ["Gaussian", "Mean"]:
            res = filters.apply_adaptive_threshold(self.image, block_size=11, c_val=2, method=method)
            self.assertEqual(res.shape, self.image.shape)

    def test_color_adjustments(self):
        b = filters.apply_brightness(self.image, brightness=30)
        c = filters.apply_contrast(self.image, contrast=1.5)
        s = filters.apply_saturation(self.image, saturation=1.5)
        self.assertEqual(b.shape, self.image.shape)
        self.assertEqual(c.shape, self.image.shape)
        self.assertEqual(s.shape, self.image.shape)

    def test_stylistic_filters(self):
        neg = filters.apply_negative(self.image)
        sep = filters.apply_sepia(self.image, intensity=0.8)
        emb = filters.apply_emboss(self.image, strength=1.2)
        cart = filters.apply_cartoon(self.image, num_bilateral=2, num_colors=8)
        self.assertEqual(neg.shape, self.image.shape)
        self.assertEqual(sep.shape, self.image.shape)
        self.assertEqual(emb.shape, self.image.shape)
        self.assertEqual(cart.shape, self.image.shape)


class TestBackgroundProcessing(unittest.TestCase):
    def setUp(self):
        # Load demo portrait if available, otherwise generate synthetic
        try:
            self.image = utils.load_image("assets/portrait.jpg")
        except Exception:
            self.image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(self.image, (320, 240), 100, (200, 200, 200), -1)

    def test_background_blur_ai(self):
        res = background.apply_background_blur(self.image, blur_strength=25, use_ai=True)
        self.assertEqual(res.shape, self.image.shape)

    def test_background_blur_fallback(self):
        res = background.apply_background_blur(self.image, blur_strength=25, use_ai=False)
        self.assertEqual(res.shape, self.image.shape)

    def test_background_removal_transparent(self):
        res = background.apply_background_removal(self.image, background_type="Transparent (PNG)", use_ai=True)
        self.assertEqual(res.ndim, 3)
        self.assertEqual(res.shape[2], 4)  # Must be RGBA

    def test_background_removal_color(self):
        for bg in ["White", "Black", "Studio Grey"]:
            res = background.apply_background_removal(self.image, background_type=bg, use_ai=True)
            self.assertEqual(res.shape, self.image.shape)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((1000, 2000, 3), dtype=np.uint8)

    def test_resize_max_dim(self):
        resized = utils.resize_image_max_dim(self.image, max_dim=800)
        self.assertEqual(max(resized.shape[:2]), 800)
        # Verify aspect ratio preserved (2000:1000 -> 2:1 -> 800:400)
        self.assertEqual(resized.shape[1], 800)
        self.assertEqual(resized.shape[0], 400)

    def test_encoding(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        jpg_bytes = utils.convert_to_download_bytes(img, format="JPEG")
        self.assertGreater(len(jpg_bytes), 0)
        png_bytes = utils.convert_to_download_bytes(img, format="PNG")
        self.assertGreater(len(png_bytes), 0)

        # RGBA encoding test
        rgba = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        rgba_png = utils.convert_to_download_bytes(rgba, format="PNG")
        self.assertGreater(len(rgba_png), 0)


def run_benchmark():
    print("\n" + "=" * 50)
    print("  PERFORMANCE BENCHMARK (30+ FPS Target Check)  ")
    print("=" * 50)
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    benchmarks = [
        ("Original", lambda: filters.apply_original(img)),
        ("Grayscale", lambda: filters.apply_grayscale(img)),
        ("Gaussian Blur (15x15)", lambda: filters.apply_gaussian_blur(img, 15)),
        ("Median Blur (11x11)", lambda: filters.apply_median_blur(img, 11)),
        ("Sharpen", lambda: filters.apply_sharpen(img, 1.5)),
        ("Canny Edge", lambda: filters.apply_canny(img, 50, 150)),
        ("Sobel Edge", lambda: filters.apply_edge_detection(img, "Sobel")),
        ("Threshold", lambda: filters.apply_threshold(img, 127)),
        ("Brightness", lambda: filters.apply_brightness(img, 30)),
        ("Contrast", lambda: filters.apply_contrast(img, 1.5)),
        ("Saturation", lambda: filters.apply_saturation(img, 1.5)),
        ("Negative", lambda: filters.apply_negative(img)),
        ("Sepia", lambda: filters.apply_sepia(img, 1.0)),
        ("Emboss", lambda: filters.apply_emboss(img, 1.0)),
    ]

    for name, fn in benchmarks:
        # Warmup
        for _ in range(3):
            fn()
        # Measure
        iterations = 50
        t0 = time.perf_counter()
        for _ in range(iterations):
            fn()
        total_time = time.perf_counter() - t0
        avg_ms = (total_time / iterations) * 1000.0
        fps = iterations / total_time
        print(f"  {name:<24}: {avg_ms:6.2f} ms | {fps:7.1f} FPS")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_benchmark()
    unittest.main()
