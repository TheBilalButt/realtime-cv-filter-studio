"""
Real Time Computer Vision Filter Studio
Built by Bilal Butt

High-performance, interactive image and video processing application.
Engineered with OpenCV, NumPy, PyTorch LRASPP MobileNet, and Streamlit.
"""

import time
import os
import threading
import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    import av
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

import filters
import background
import utils

# --- Thread-Safe Holder for Live WebRTC Processing ---
class LiveFilterHolder:
    _lock = threading.Lock()
    _filter_name = "Original"
    _params = {}

    @classmethod
    def set(cls, filter_name, params):
        with cls._lock:
            cls._filter_name = filter_name
            cls._params = params

    @classmethod
    def get(cls):
        with cls._lock:
            return cls._filter_name, cls._params


# STUN servers for NAT traversal on remote hosts
RTC_CONFIGURATION = (
    RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
            ]
        }
    )
    if HAS_WEBRTC
    else None
)

# --- Page Configuration (No emojis) ---
st.set_page_config(
    page_title="Real Time CV Filter Studio - Bilal Butt",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Clean Professional Dark Studio Styling ---
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">

    <style>
    /* Global Reset & Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #0b0f19 !important;
        color: #f1f5f9 !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1360px !important;
    }

    /* Header Styling */
    .studio-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 1rem;
        margin-bottom: 1.25rem;
        flex-wrap: wrap;
        gap: 12px;
    }
    .studio-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0;
    }
    .studio-subtitle {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    .author-badge {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 6px 14px;
        border-radius: 6px;
    }

    /* Stage Panel & Labels */
    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #111827;
        border: 1px solid #1f2937;
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        padding: 8px 14px;
    }
    .panel-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
    }
    .panel-status {
        font-size: 0.72rem;
        font-weight: 600;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        padding: 2px 8px;
        border-radius: 4px;
    }

    /* Image Wrapper */
    .image-frame {
        border: 1px solid #1f2937;
        border-radius: 0 0 8px 8px;
        background: #030712;
        overflow: hidden;
        margin-bottom: 0.75rem;
    }

    /* Performance Metric Cards */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .metric-number {
        font-family: 'Space Grotesk', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-title {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-top: 2px;
    }

    /* Filter Cards & Thumbnails */
    .filter-card {
        border: 1px solid #1e293b;
        border-radius: 8px;
        background: #0f172a;
        padding: 6px;
        text-align: center;
        transition: all 0.2s ease;
        margin-bottom: 8px;
    }
    .filter-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .filter-card.active {
        border: 2px solid #38bdf8 !important;
        background: rgba(56, 189, 248, 0.1) !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25);
    }
    .filter-title {
        font-size: 0.76rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Control Box */
    .controls-container {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Action Buttons */
    div.stButton > button {
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border: 1px solid #334155 !important;
        background: #1e293b !important;
        color: #f1f5f9 !important;
        transition: all 0.15s ease !important;
        padding: 6px 14px !important;
    }
    div.stButton > button:hover {
        background: #2563eb !important;
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* Active Filter Button Highlight */
    .active-filter-btn button {
        background: #2563eb !important;
        border-color: #60a5fa !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.4) !important;
    }

    /* Download Button */
    .stDownloadButton button {
        background: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: 1px solid #0369a1 !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton button:hover {
        background: #0369a1 !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #111827;
        padding: 4px;
        border-radius: 8px;
        border: 1px solid #1f2937;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        padding: 6px 18px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #94a3b8 !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        color: #38bdf8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Filter definitions with short technical description
FILTERS_CATALOG = [
    {
        "id": "Original",
        "name": "Original",
        "desc": "Unprocessed reference source input.",
    },
    {
        "id": "Grayscale",
        "name": "Grayscale",
        "desc": "Rec. 601 single-channel luminance conversion.",
    },
    {
        "id": "Gaussian Blur",
        "name": "Gaussian Blur",
        "desc": "2D Gaussian kernel low-pass spatial smoothing.",
    },
    {
        "id": "Median Blur",
        "name": "Median Blur",
        "desc": "Non-linear filter removing impulse and salt-pepper noise.",
    },
    {
        "id": "Sharpen",
        "name": "Sharpen",
        "desc": "High-frequency unsharp masking enhancement.",
    },
    {
        "id": "Edge Detection",
        "name": "Edge Detection",
        "desc": "Spatial gradient edge detection via Sobel / Laplacian.",
    },
    {
        "id": "Canny Edge Detection",
        "name": "Canny Edge",
        "desc": "Multi-stage optimal hysteresis edge detector.",
    },
    {
        "id": "Threshold",
        "name": "Threshold",
        "desc": "Intensity segmentation (Binary or automated Otsu).",
    },
    {
        "id": "Adaptive Threshold",
        "name": "Adaptive Thresh",
        "desc": "Local neighborhood adaptive thresholding.",
    },
    {
        "id": "Brightness",
        "name": "Brightness",
        "desc": "Linear luminance offset across color channels.",
    },
    {
        "id": "Contrast",
        "name": "Contrast",
        "desc": "Luminance scaling and dynamic range multiplier.",
    },
    {
        "id": "Saturation",
        "name": "Saturation",
        "desc": "HSV color space chromatic intensity scale.",
    },
    {
        "id": "Negative",
        "name": "Negative",
        "desc": "Bitwise inversion of color intensity channels.",
    },
    {
        "id": "Sepia",
        "name": "Sepia",
        "desc": "Three-channel photographic warm sepia matrix transform.",
    },
    {
        "id": "Emboss",
        "name": "Emboss",
        "desc": "Directional relief convolution matrix with 128 bias.",
    },
    {
        "id": "Cartoon Effect",
        "name": "Cartoon",
        "desc": "Edge-preserving bilateral filter with color quantization.",
    },
    {
        "id": "Background Blur",
        "name": "Background Blur",
        "desc": "Foreground segmentation with background depth bokeh blur.",
    },
    {
        "id": "Background Removal",
        "name": "Background Removal",
        "desc": "Alpha extraction producing clean transparent PNG cutout.",
    },
]

FILTER_NAMES = [f["id"] for f in FILTERS_CATALOG]


@st.cache_resource
def load_segmentation_model():
    """Initializes segmentation model in memory."""
    try:
        return background.get_segmentation_model()
    except Exception:
        return None


# Pre-warm model in background
_ = load_segmentation_model()


def get_default_image() -> np.ndarray:
    """Loads default reference portrait image from assets."""
    path = os.path.join(os.path.dirname(__file__), "assets", "portrait.jpg")
    if os.path.exists(path):
        return utils.load_image(path)
    # Synthetic fallback
    synth = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(synth, (320, 240), 120, (200, 180, 160), -1)
    return synth


def apply_filter_pipeline(image_rgb: np.ndarray, filter_name: str, params: dict):
    """Executes filter and tracks runtime latency and FPS."""
    t0 = time.perf_counter()

    if filter_name == "Original":
        processed = filters.apply_original(image_rgb)
    elif filter_name == "Grayscale":
        processed = filters.apply_grayscale(image_rgb)
    elif filter_name == "Gaussian Blur":
        processed = filters.apply_gaussian_blur(
            image_rgb,
            kernel_size=params.get("kernel_size", 15),
            sigma=params.get("sigma", 0.0),
        )
    elif filter_name == "Median Blur":
        processed = filters.apply_median_blur(
            image_rgb,
            kernel_size=params.get("kernel_size", 11),
        )
    elif filter_name == "Sharpen":
        processed = filters.apply_sharpen(
            image_rgb,
            strength=params.get("strength", 1.5),
        )
    elif filter_name == "Edge Detection":
        processed = filters.apply_edge_detection(
            image_rgb,
            method=params.get("method", "Sobel"),
            kernel_size=params.get("kernel_size", 3),
        )
    elif filter_name == "Canny Edge Detection":
        processed = filters.apply_canny(
            image_rgb,
            low_threshold=params.get("low_threshold", 50),
            high_threshold=params.get("high_threshold", 150),
        )
    elif filter_name == "Threshold":
        processed = filters.apply_threshold(
            image_rgb,
            thresh=params.get("thresh", 127),
            max_val=params.get("max_val", 255),
            threshold_type=params.get("threshold_type", "Binary"),
        )
    elif filter_name == "Adaptive Threshold":
        processed = filters.apply_adaptive_threshold(
            image_rgb,
            max_val=params.get("max_val", 255),
            block_size=params.get("block_size", 11),
            c_val=params.get("c_val", 2),
            method=params.get("method", "Gaussian"),
        )
    elif filter_name == "Brightness":
        processed = filters.apply_brightness(
            image_rgb,
            brightness=params.get("brightness", 30),
        )
    elif filter_name == "Contrast":
        processed = filters.apply_contrast(
            image_rgb,
            contrast=params.get("contrast", 1.5),
        )
    elif filter_name == "Saturation":
        processed = filters.apply_saturation(
            image_rgb,
            saturation=params.get("saturation", 1.5),
        )
    elif filter_name == "Negative":
        processed = filters.apply_negative(image_rgb)
    elif filter_name == "Sepia":
        processed = filters.apply_sepia(
            image_rgb,
            intensity=params.get("intensity", 1.0),
        )
    elif filter_name == "Emboss":
        processed = filters.apply_emboss(
            image_rgb,
            strength=params.get("strength", 1.0),
        )
    elif filter_name == "Cartoon Effect":
        processed = filters.apply_cartoon(
            image_rgb,
            num_bilateral=params.get("num_bilateral", 2),
            num_colors=params.get("num_colors", 8),
            edge_kernel=params.get("edge_kernel", 7),
        )
    elif filter_name == "Background Blur":
        processed = background.apply_background_blur(
            image_rgb,
            blur_strength=params.get("blur_strength", 35),
            feather=params.get("feather", 5),
            threshold=params.get("threshold", 0.4),
            use_ai=params.get("use_ai", True),
        )
    elif filter_name == "Background Removal":
        processed = background.apply_background_removal(
            image_rgb,
            background_type=params.get("background_type", "Transparent (PNG)"),
            custom_color=params.get("custom_color", (255, 255, 255)),
            feather=params.get("feather", 5),
            threshold=params.get("threshold", 0.4),
            use_ai=params.get("use_ai", True),
        )
    else:
        processed = image_rgb.copy()

    elapsed = time.perf_counter() - t0
    latency_ms = elapsed * 1000.0
    fps = (1.0 / elapsed) if elapsed > 0 else 0.0

    return processed, latency_ms, fps


def webrtc_video_frame_callback(frame: "av.VideoFrame") -> "av.VideoFrame":
    """Real-time frame processing callback for WebRTC live camera."""
    img_bgr = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    cur_filter, cur_params = LiveFilterHolder.get()
    try:
        processed, _, _ = apply_filter_pipeline(img_rgb, cur_filter, cur_params)
        if processed.ndim == 3 and processed.shape[2] == 4:
            alpha = processed[:, :, 3:4] / 255.0
            fg = processed[:, :, :3]
            bg = np.full_like(fg, 20)
            comp = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
            out_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
        else:
            out_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    except Exception:
        out_bgr = img_bgr

    return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")


def render_filter_thumbnails_bar(base_image: np.ndarray, active_filter: str):
    """
    Generates miniature previews for the key filters and displays an interactive gallery.
    Clicking any card selects the filter immediately.
    """
    # Downscale to 80px thumbnail for sub-millisecond generation
    h, w = base_image.shape[:2]
    thumb_w = 90
    thumb_h = max(1, int(round(thumb_w * h / w)))
    thumb = cv2.resize(base_image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    # Primary gallery filters to showcase
    gallery_keys = [
        ("Original", "Original", lambda img: filters.apply_original(img)),
        ("Cartoon Effect", "Cartoon", lambda img: filters.apply_cartoon(img, 1, 8, 5)),
        ("Background Blur", "BG Blur", lambda img: filters.apply_background_blur(img, 25, 3, 0.4, False)),
        ("Background Removal", "Cutout", lambda img: background.apply_background_removal(img, "Studio Grey", (200, 200, 200), 3, 0.4, False)),
        ("Grayscale", "Grayscale", lambda img: filters.apply_grayscale(img)),
        ("Sharpen", "Sharpen", lambda img: filters.apply_sharpen(img, 2.0)),
        ("Canny Edge Detection", "Canny", lambda img: filters.apply_canny(img, 50, 150)),
        ("Sepia", "Sepia", lambda img: filters.apply_sepia(img, 1.0)),
        ("Emboss", "Emboss", lambda img: filters.apply_emboss(img, 1.2)),
        ("Threshold", "Threshold", lambda img: filters.apply_threshold(img, 127)),
    ]

    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 8px;'>Filter Previews (Click to apply)</div>", unsafe_allow_html=True)
    
    cols = st.columns(len(gallery_keys))
    for i, (f_id, f_label, f_fn) in enumerate(gallery_keys):
        with cols[i]:
            try:
                t_proc = f_fn(thumb)
            except Exception:
                t_proc = thumb

            is_active = (active_filter == f_id)
            active_class = "active" if is_active else ""
            status_text = "[ACTIVE]" if is_active else ""

            # Card image
            st.image(t_proc, use_container_width=True)
            
            # Clickable button with active state styling
            btn_label = f"{f_label} *" if is_active else f_label
            if st.button(btn_label, key=f"thumb_btn_{f_id}", use_container_width=True):
                st.session_state["active_filter"] = f_id
                st.rerun()


def render_parameter_controls(active_filter: str) -> dict:
    """Renders parameter sliders and controls for the active filter."""
    params = {}

    st.markdown(f"<div style='font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #38bdf8; margin-bottom: 10px;'>Parameter Controls: {active_filter}</div>", unsafe_allow_html=True)

    if active_filter == "Cartoon Effect":
        c1, c2, c3 = st.columns(3)
        with c1:
            params["num_bilateral"] = st.slider("Bilateral Passes (Smoothing)", 1, 5, 2, step=1)
        with c2:
            params["num_colors"] = st.slider("Color Quantization (Levels)", 4, 32, 8, step=2)
        with c3:
            params["edge_kernel"] = st.slider("Outline Thickness", 3, 15, 7, step=2)

    elif active_filter == "Background Blur":
        c1, c2, c3 = st.columns(3)
        with c1:
            params["blur_strength"] = st.slider("Background Blur Radius", 5, 75, 35, step=2)
        with c2:
            params["feather"] = st.slider("Edge Feather Radius", 1, 15, 5, step=1)
        with c3:
            params["threshold"] = st.slider("Detection Threshold", 0.1, 0.9, 0.4, step=0.05)
            params["use_ai"] = st.checkbox("AI LRASPP Segmentation", value=True)

    elif active_filter == "Background Removal":
        c1, c2, c3 = st.columns(3)
        with c1:
            params["background_type"] = st.selectbox(
                "Replacement Background:",
                ["Transparent (PNG)", "White", "Black", "Studio Grey", "Custom Color"],
            )
            if params["background_type"] == "Custom Color":
                col = st.color_picker("Color Value", "#38BDF8")
                params["custom_color"] = (int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16))
            else:
                params["custom_color"] = (255, 255, 255)
        with c2:
            params["feather"] = st.slider("Edge Feather Radius", 1, 15, 5, step=1)
        with c3:
            params["threshold"] = st.slider("Detection Threshold", 0.1, 0.9, 0.4, step=0.05)
            params["use_ai"] = st.checkbox("AI LRASPP Segmentation", value=True)

    elif active_filter == "Sharpen":
        params["strength"] = st.slider("Sharpening Strength", 0.1, 4.0, 1.5, step=0.1)

    elif active_filter == "Gaussian Blur":
        c1, c2 = st.columns(2)
        with c1:
            params["kernel_size"] = st.slider("Kernel Size (Odd)", 3, 51, 15, step=2)
        with c2:
            params["sigma"] = st.slider("Sigma (Dispersion)", 0.0, 10.0, 0.0, step=0.5)

    elif active_filter == "Median Blur":
        params["kernel_size"] = st.slider("Kernel Size (Odd)", 3, 45, 11, step=2)

    elif active_filter == "Brightness":
        params["brightness"] = st.slider("Brightness Offset", -100, 100, 30, step=5)

    elif active_filter == "Contrast":
        params["contrast"] = st.slider("Contrast Factor", 0.1, 3.0, 1.5, step=0.1)

    elif active_filter == "Saturation":
        params["saturation"] = st.slider("Saturation Multiplier", 0.0, 3.0, 1.5, step=0.1)

    elif active_filter == "Sepia":
        params["intensity"] = st.slider("Sepia Intensity", 0.0, 1.0, 1.0, step=0.05)

    elif active_filter == "Emboss":
        params["strength"] = st.slider("Relief Depth", 0.2, 3.0, 1.0, step=0.1)

    elif active_filter == "Canny Edge Detection":
        c1, c2 = st.columns(2)
        with c1:
            params["low_threshold"] = st.slider("Low Hysteresis Threshold", 0, 255, 50, step=5)
        with c2:
            params["high_threshold"] = st.slider("High Hysteresis Threshold", 0, 255, 150, step=5)

    elif active_filter == "Edge Detection":
        c1, c2 = st.columns(2)
        with c1:
            params["method"] = st.selectbox("Operator", ["Sobel", "Laplacian", "Prewitt"])
        with c2:
            params["kernel_size"] = st.slider("Kernel Size", 1, 7, 3, step=2)

    elif active_filter == "Threshold":
        c1, c2 = st.columns(2)
        with c1:
            params["threshold_type"] = st.selectbox("Method", ["Binary", "Binary Inverted", "Otsu"])
        with c2:
            if params["threshold_type"] != "Otsu":
                params["thresh"] = st.slider("Cutoff Level", 0, 255, 127, step=1)
            else:
                params["thresh"] = 0
                st.caption("Otsu algorithm calculates threshold automatically.")
            params["max_val"] = 255

    elif active_filter == "Adaptive Threshold":
        c1, c2 = st.columns(2)
        with c1:
            params["method"] = st.selectbox("Method", ["Gaussian", "Mean"])
            params["block_size"] = st.slider("Neighborhood Size", 3, 51, 11, step=2)
        with c2:
            params["c_val"] = st.slider("Constant C", -20, 20, 2, step=1)
            params["max_val"] = 255

    else:
        st.caption("This filter executes with standardized optimal parameters.")

    return params


def main():
    # Session state initialization
    if "active_filter" not in st.session_state:
        st.session_state["active_filter"] = "Cartoon Effect"

    if "current_image" not in st.session_state:
        st.session_state["current_image"] = get_default_image()

    # --- Header Bar ---
    st.markdown(
        """
        <div class="studio-header">
            <div>
                <h1 class="studio-title">Real Time Computer Vision Filter Studio</h1>
                <div class="studio-subtitle">Interactive high-performance computer vision image and video processing</div>
            </div>
            <div class="author-badge">Built by Bilal Butt</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Tabs
    tab_image_studio, tab_live_camera = st.tabs(["Image Studio", "Live Camera Stream"])

    # =========================================================================
    # TAB 1: IMAGE STUDIO (Original Image | Processed Image Side-by-Side)
    # =========================================================================
    with tab_image_studio:
        # Input Controls Bar
        with st.expander("Input Source & Image Selection", expanded=False):
            c_src1, c_src2, c_src3 = st.columns([2, 2, 1])
            with c_src1:
                uploaded = st.file_uploader(
                    "Upload Image (JPG, PNG, WEBP):",
                    type=["jpg", "jpeg", "png", "webp", "bmp"],
                )
                if uploaded is not None:
                    try:
                        st.session_state["current_image"] = utils.load_image(uploaded)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")

            with c_src2:
                sample_options = {
                    "Portrait Demo (AI Background)": "portrait.jpg",
                    "Landscape Demo (Textures & Edges)": "landscape.jpg",
                }
                chosen_sample = st.selectbox("Or Choose Preset Sample:", list(sample_options.keys()))
                if st.button("Load Selected Sample", use_container_width=True):
                    path = os.path.join(os.path.dirname(__file__), "assets", sample_options[chosen_sample])
                    if os.path.exists(path):
                        st.session_state["current_image"] = utils.load_image(path)
                        st.rerun()

            with c_src3:
                snap = st.camera_input("Quick Camera Snapshot")
                if snap is not None:
                    try:
                        st.session_state["current_image"] = utils.load_image(snap)
                    except Exception as e:
                        st.error(f"Snapshot error: {e}")

        # Active Image Reference
        input_img = st.session_state["current_image"]
        active_filter = st.session_state["active_filter"]

        # 1. Filter Thumbnails Quick-Bar
        render_filter_thumbnails_bar(input_img, active_filter)

        # 2. Filter Dropdown for All 18 Filters
        all_col1, all_col2 = st.columns([3, 1])
        with all_col1:
            selected_from_menu = st.selectbox(
                "Filter Selection:",
                FILTER_NAMES,
                index=FILTER_NAMES.index(active_filter) if active_filter in FILTER_NAMES else 0,
                key="filter_menu_select",
            )
            if selected_from_menu != active_filter:
                st.session_state["active_filter"] = selected_from_menu
                st.rerun()

        with all_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Reset to Original", use_container_width=True):
                st.session_state["active_filter"] = "Original"
                st.rerun()

        # 3. Dynamic Parameter Controls
        with st.expander(f"Adjust Parameters: {active_filter}", expanded=True):
            params = render_parameter_controls(active_filter)

        # Sync thread-safe holder for WebRTC
        LiveFilterHolder.set(active_filter, params)

        # 4. Process the Active Image
        # Scale for optimal real-time responsiveness if exceeds 1080p
        scaled_input = utils.resize_image_max_dim(input_img, max_dim=1080)
        h, w = scaled_input.shape[:2]

        processed_result, latency_ms, fps = apply_filter_pipeline(scaled_input, active_filter, params)
        is_rgba = (processed_result.ndim == 3 and processed_result.shape[2] == 4)

        # 5. Core View: ORIGINAL IMAGE | PROCESSED IMAGE (50/50 Dual Display)
        st.markdown("<hr style='margin: 16px 0; opacity: 0.15;'>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(
                """
                <div class="panel-header">
                    <span class="panel-label">Original Image</span>
                    <span class="panel-status">Reference</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div class='image-frame'>", unsafe_allow_html=True)
            st.image(scaled_input, channels="RGB", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.caption(f"Dimensions: {w} x {h} px | Channels: 3 (RGB)")

        with col_right:
            st.markdown(
                f"""
                <div class="panel-header">
                    <span class="panel-label">Processed Image: {active_filter}</span>
                    <span class="panel-status">[ACTIVE]</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div class='image-frame'>", unsafe_allow_html=True)
            st.image(
                processed_result,
                channels="RGBA" if is_rgba else "RGB",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Real-time Metrics Dashboard
            st.markdown(
                f"""
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-number">{latency_ms:.1f} ms</div>
                        <div class="metric-title">Latency</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-number">{fps:.0f}</div>
                        <div class="metric-title">FPS</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-number">{w}x{h}</div>
                        <div class="metric-title">Resolution</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-number">{'4 (RGBA)' if is_rgba else '3 (RGB)'}</div>
                        <div class="metric-title">Channels</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Download Button
            if is_rgba:
                fmt, mime, ext = "PNG", "image/png", "png"
            else:
                fmt, mime, ext = "JPEG", "image/jpeg", "jpg"

            export_bytes = utils.convert_to_download_bytes(processed_result, format=fmt)
            filename = f"filtered_{active_filter.lower().replace(' ', '_')}.{ext}"

            st.download_button(
                label=f"Download Processed Image ({fmt} - {len(export_bytes) // 1024} KB)",
                data=export_bytes,
                file_name=filename,
                mime=mime,
                use_container_width=True,
            )

    # =========================================================================
    # TAB 2: LIVE CAMERA STREAM (Browser WebRTC)
    # =========================================================================
    with tab_live_camera:
        st.markdown(
            f"""
            <div class="panel-header" style="margin-bottom: 12px; border-radius: 8px;">
                <span class="panel-label">Real-Time Browser Camera Stream</span>
                <span class="panel-status">Active Filter: {active_filter}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("Click **START** below and allow camera permissions in your browser. Video is processed frame-by-frame using the active filter.")

        if HAS_WEBRTC:
            webrtc_streamer(
                key="studio-live-stream-streamer",
                video_frame_callback=webrtc_video_frame_callback,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                async_processing=True,
            )
        else:
            st.error("WebRTC streaming module is not available in the current environment.")


if __name__ == "__main__":
    main()
