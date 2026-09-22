# Real Time Computer Vision Filter Studio

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LRASPP%20MobileNet-EE4C2C.svg)](https://pytorch.org/)
[![Deploy with Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=TheBilalButt/realtime-cv-filter-studio&branch=main&mainModule=app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Built by Bilal Butt**  
> A fast, responsive, and lightweight real-time Computer Vision filter studio built with Python, OpenCV, NumPy, and Streamlit.  
> Public Access Link: [Deploy / Launch on Streamlit Community Cloud](https://share.streamlit.io/deploy?repository=TheBilalButt/realtime-cv-filter-studio&branch=main&mainModule=app.py)

---

## Project Overview

Real Time Computer Vision Filter Studio is a high-performance web application designed for interactive image and video processing. It allows users to apply 18 distinct computer vision filters, tune algorithm parameters dynamically with live sliders, compare before/after images side-by-side in real time, and inspect processing latency and FPS metrics.

The system is optimized for speed on consumer laptops, leveraging vectorized NumPy/OpenCV routines and a cached lightweight deep learning model (LRASPP MobileNetV3) for real-time background segmentation and bokeh simulation.

---

## Features

- Multiple Media Input Sources:
  - Image upload (JPG, PNG, WEBP, BMP)
  - Webcam snapshot capture using native browser camera integration
  - Live real-time webcam video stream loop via WebRTC with live FPS tracking
  - Built-in preset sample images (Portrait and Landscape) for instant testing
- 18 High-Performance Computer Vision Filters:
  - Classic spatial filters, edge operators, color adjustments, artistic effects, and AI segmentation
- Interactive Dynamic Parameters:
  - Real-time sliders for kernel sizes, sigma, thresholds, brightness, contrast, saturation, and feathering
- Side-by-Side Real-Time Comparison:
  - Original Image displayed adjacent to Processed Image with immediate parameter updates
- Hardware-Accelerated Background Processing:
  - AI-based foreground segmentation using LRASPP MobileNetV3 (~12MB)
  - Cached foreground alpha mask ensuring sub-millisecond slider responsiveness
  - Ultra-fast OpenCV GrabCut fallback for instant execution
  - Background Bokeh Blur with adjustable radius and edge feathering
  - Background Removal with transparent PNG export or solid studio backdrop replacement
- Real-Time Performance Dashboard:
  - Millisecond latency measurement
  - Effective Frames Per Second (FPS) calculation
  - Image resolution indicator with automated high-resolution downscaling
- Clean Export:
  - Download processed images (PNG with alpha transparency for cutouts, high-quality JPEG for standard filters)

---

## User Interface Layout

```text
+---------------------------------------------------------------------------------------+
|  Real Time Computer Vision Filter Studio                         [Built by Bilal Butt]|
+------------------------------------+--------------------------------------------------+
|  FILTER SELECTION & CONTROLS       |  MAIN WORKSPACE (50/50 DUAL DISPLAY)             |
|  ---------------------------       |  -----------------------------------             |
|  Filter Thumbnails Gallery         |  [ Latency: 2.1 ms ]  [ FPS: 480 ]  [ 800x600 ]  |
|  [Cartoon] [Blur] [Cutout] [B&W]   |                                                  |
|                                    |  +---------------------+  +--------------------+ |
|  Parameter Sliders:                |  |   ORIGINAL IMAGE    |  |  PROCESSED RESULT  | |
|  - Slider 1: [---o-----]           |  |                     |  |  (Updates in       | |
|  - Slider 2: [------o--]           |  |                     |  |   real time)       | |
|                                    |  +---------------------+  +--------------------+ |
|                                    |                                                  |
|                                    |  [ Download Processed Image (PNG/JPEG) ]         |
+------------------------------------+--------------------------------------------------+
```

---

## Tech Stack

- Language: Python 3.11+ (Tested on Python 3.12 and 3.13)
- Computer Vision: OpenCV (cv2) and NumPy (Vectorized C-backend operations)
- Web UI: Streamlit
- Live WebRTC: streamlit-webrtc and PyAV
- Deep Learning: PyTorch and Torchvision (LRASPP MobileNetV3 Large)
- Image I/O: Pillow (PIL)

---

## Project Architecture

```text
realtime-cv-filter-studio/
|
|-- app.py              # Streamlit web application and UI layout
|-- filters.py          # Vectorized OpenCV and NumPy filter implementations
|-- background.py       # AI semantic segmentation and GrabCut fallback engine
|-- utils.py            # Image loader, aspect-ratio resizer, metrics and export
|-- create_assets.py    # Generator for sample testing assets
|-- test_suite.py       # Automated unit test and FPS benchmark suite
|-- requirements.txt    # Application dependencies
|-- packages.txt        # System packages for headless Linux deployment
|-- .gitignore          # Git exclusion rules
|-- assets/             # Demo images and studio assets
|   |-- portrait.jpg
|   `-- landscape.jpg
`-- README.md           # Documentation and setup instructions
```

---

## Available Filters

| Category | Filter Name | Key Parameters | Typical FPS (Laptop CPU) |
| :--- | :--- | :--- | :--- |
| **Identity & Base** | Original | N/A | **10,000+ FPS** |
| **Color Spaces** | Grayscale | N/A | **4,800+ FPS** |
| | Negative | N/A | **5,700+ FPS** |
| | Sepia | Tone Intensity (0.0 to 1.0) | **600+ FPS** |
| | Brightness | Offset (-100 to +100) | **3,400+ FPS** |
| | Contrast | Multiplier (0.1 to 3.0) | **3,700+ FPS** |
| | Saturation | Multiplier (0.0 to 3.0) | **110+ FPS** |
| **Smoothing** | Gaussian Blur | Kernel Size (3-51), Sigma | **550+ FPS** |
| | Median Blur | Kernel Size (3-45) | **15-30+ FPS** |
| **Detail & Edges** | Sharpen | Strength (0.1 to 4.0) | **280+ FPS** |
| | Edge Detection | Algorithm (Sobel, Laplacian, Prewitt), Kernel | **800+ FPS** |
| | Canny Edge Detection | Low & High Hysteresis Thresholds | **220+ FPS** |
| **Thresholding** | Threshold | Binary, Binary Inverted, Otsu | **7,200+ FPS** |
| | Adaptive Threshold | Gaussian / Mean, Block Size, C Constant | **350+ FPS** |
| **Stylization** | Emboss | 3D Relief Strength (0.2 to 3.0) | **600+ FPS** |
| | Cartoon Effect | Bilateral Passes, Color Palette, Edge Kernel | **30+ FPS** |
| **Segmentation** | Background Blur | Blur Radius, Feathering, AI / Fast Fallback | **Real-Time (~150ms AI / ~25ms Fast)** |
| | Background Removal | Transparent PNG, Studio Backdrops, Feathering | **Real-Time (~150ms AI / ~30ms Fast)** |

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/TheBilalButt/realtime-cv-filter-studio.git
cd realtime-cv-filter-studio
```

### 2. Set up a virtual environment (Recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501` in your browser.

---

## Running Automated Tests & Benchmarks

Run the complete automated unit test suite and FPS benchmark:

```bash
python test_suite.py
```

All 17 tests verify matrix bounds, channel preservation, alpha composition, and execution speed.

---

## Performance & Design Principles

1. No Slow Python Loops: Every pixel manipulation utilizes OpenCV C-extensions and NumPy vectorized array operations.
2. Resource and Mask Caching: The PyTorch neural segmentation model and computed masks are cached, preventing repetitive neural network passes during parameter adjustments.
3. Adaptive Downscaling: High-resolution inputs are resized to an optimal processing ceiling (1080p by default) while maintaining aspect ratio, keeping operations smooth and preventing memory exhaustion.
4. Seamless Fallback: If deep learning segmentation is toggled off or requires extreme low-power efficiency, an OpenCV GrabCut engine provides near-instantaneous fallback.

---

## Author

**Built by Bilal Butt**  
- GitHub: https://github.com/TheBilalButt
- Project: Real Time Computer Vision Filter Studio

---

## License

This project is open source and available under the [MIT License](LICENSE).
