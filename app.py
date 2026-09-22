"""
Real Time Computer Vision Filter Studio
Built by Bilal Butt

A modern, user-friendly, high-performance Computer Vision studio.
Accessible to everyone — from complete beginners to CV professionals!
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


# Google STUN servers for NAT/firewall traversal on Streamlit Cloud
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

# --- Page Configuration ---
st.set_page_config(
    page_title="Filter Studio | Bilal Butt",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling with Google Fonts & Polished Glassmorphism ---
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">

    <style>
    /* Global font & typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Headings */
    h1, h2, h3, .main-title {
        font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: -0.02em;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }

    .author-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.15));
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        color: #38bdf8;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
        margin-bottom: 0.8rem;
    }

    .lead-text {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }

    /* Step helper box */
    .step-box {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 0.85rem 1.25rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
    }
    .step-pill {
        font-weight: 700;
        font-size: 0.9rem;
        color: #e2e8f0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc;
        font-family: 'Space Grotesk', sans-serif;
    }
    .metric-lbl {
        font-size: 0.78rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }

    /* Description pill */
    .filter-desc-box {
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8));
        border-left: 4px solid #818cf8;
        border-radius: 0 10px 10px 0;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1.2rem;
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    /* Big Friendly Buttons */
    div.stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.55rem 1.25rem !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(56, 189, 248, 0.25) !important;
    }

    /* Primary Download Button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
        opacity: 0.96;
    }

    /* Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 8px 20px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(129, 140, 248, 0.2)) !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_cached_segmentation_model():
    """Caches the PyTorch LRASPP segmentation model in memory."""
    try:
        return background.get_segmentation_model()
    except Exception:
        return None


# Pre-warm model cache silently
_ = load_cached_segmentation_model()

# Dictionary of friendly explanations for all filters
FILTER_INFO = {
    "Original": {
        "icon": "📷",
        "title": "Original Photo",
        "desc": "Shows your natural photo or video stream without any changes.",
    },
    "Cartoon Effect": {
        "icon": "🎨",
        "title": "Cartoon / Anime Style",
        "desc": "Flattens colors and draws bold outline edges like a colorful animated movie!",
    },
    "Background Blur": {
        "icon": "🌫️",
        "title": "DSLR Portrait Bokeh",
        "desc": "Keeps you crystal sharp while beautifully blurring the background like a professional camera lens.",
    },
    "Background Removal": {
        "icon": "✂️",
        "title": "Transparent Cutout Sticker",
        "desc": "Removes the background completely so you can save as a transparent PNG or put yourself in a clean photo studio.",
    },
    "Grayscale": {
        "icon": "🖤",
        "title": "Classic Black & White",
        "desc": "Converts color into timeless monochrome studio photography.",
    },
    "Sharpen": {
        "icon": "🗡️",
        "title": "Ultra Crisp Sharpen",
        "desc": "Enhances fine details, hair, and edges for maximum punch and crispness.",
    },
    "Canny Edge Detection": {
        "icon": "✏️",
        "title": "Pencil Line Sketch",
        "desc": "Extracts the key contours and outlines like an artist's pencil drawing.",
    },
    "Edge Detection": {
        "icon": "⚡",
        "title": "Neon Contour Glow",
        "desc": "Calculates directional gradients (Sobel / Laplacian) across every boundary.",
    },
    "Gaussian Blur": {
        "icon": "💨",
        "title": "Smooth Gaussian Softener",
        "desc": "Softens the entire image to smooth skin or create a dreamy atmosphere.",
    },
    "Median Blur": {
        "icon": "🫧",
        "title": "Noise Cleaning Blur",
        "desc": "Specialized filter that eliminates grain and speckles while preserving sharp outlines.",
    },
    "Brightness": {
        "icon": "☀️",
        "title": "Sunlight Brightness",
        "desc": "Easily lightens or dims your photo to fix dark or overexposed lighting.",
    },
    "Contrast": {
        "icon": "🌗",
        "title": "Punchy Contrast",
        "desc": "Deepens shadows and elevates highlights for a cinematic, dramatic punch.",
    },
    "Saturation": {
        "icon": "🌈",
        "title": "Vibrant Color Boost",
        "desc": "Amplifies color richness from gentle pastels to vivid, punchy pop-art.",
    },
    "Negative": {
        "icon": "🔄",
        "title": "X-Ray Negative",
        "desc": "Inverts all colors for an eerie sci-fi film negative effect.",
    },
    "Sepia": {
        "icon": "📜",
        "title": "1970s Vintage Retro",
        "desc": "Applies a warm, nostalgic antique tone like a precious historic photograph.",
    },
    "Emboss": {
        "icon": "🗿",
        "title": "3D Carved Metal",
        "desc": "Transforms your image into an embossed 3D stone or metallic engraving.",
    },
    "Threshold": {
        "icon": "🏁",
        "title": "High-Contrast Binary",
        "desc": "Forces every pixel to either pure black or pure white (stamp / stencil look).",
    },
    "Adaptive Threshold": {
        "icon": "📑",
        "title": "Document Scanner",
        "desc": "Compensates for uneven shadows, perfect for text, document scanning, and comic inks.",
    },
}

ALL_FILTER_NAMES = list(FILTER_INFO.keys())


def get_available_presets():
    """Returns sample preset images."""
    presets = {}
    sample_dir = os.path.join(os.path.dirname(__file__), "assets")
    if os.path.exists(os.path.join(sample_dir, "portrait.jpg")):
        presets["👤 Portrait Model (AI Background Demo)"] = os.path.join(sample_dir, "portrait.jpg")
    if os.path.exists(os.path.join(sample_dir, "landscape.jpg")):
        presets["🌄 Scenic Landscape (Colors & Edges Demo)"] = os.path.join(sample_dir, "landscape.jpg")
    return presets


def process_image(image_rgb: np.ndarray, filter_name: str, params: dict):
    """Applies filter and computes execution duration and FPS."""
    t_start = time.perf_counter()

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
            num_bilateral=params.get("num_bilateral", 3),
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

    duration_sec = time.perf_counter() - t_start
    duration_ms = duration_sec * 1000.0
    fps = (1.0 / duration_sec) if duration_sec > 0 else 0.0

    return processed, duration_ms, fps


def webrtc_video_frame_callback(frame: "av.VideoFrame") -> "av.VideoFrame":
    """Processes live WebRTC webcam frames in real-time."""
    img_bgr = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    cur_filter, cur_params = LiveFilterHolder.get()
    try:
        processed, _, _ = process_image(img_rgb, cur_filter, cur_params)
        if processed.ndim == 3 and processed.shape[2] == 4:
            alpha = processed[:, :, 3:4] / 255.0
            fg = processed[:, :, :3]
            bg = np.full_like(fg, 25)
            comp = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
            out_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
        else:
            out_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    except Exception:
        out_bgr = img_bgr

    return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")


def render_sidebar():
    """Renders user-friendly sidebar controls."""
    if "selected_filter" not in st.session_state:
        st.session_state["selected_filter"] = "Cartoon Effect"

    with st.sidebar:
        st.markdown("<div class='author-pill'>✨ Built by Bilal Butt</div>", unsafe_allow_html=True)
        st.markdown("### 🎛️ Filter Selection")

        # Friendly quick-selection buttons
        st.markdown("**⭐ Popular Filters (Click to Apply):**")
        qcol1, qcol2 = st.columns(2)
        with qcol1:
            if st.button("🎨 Cartoon", use_container_width=True):
                st.session_state["selected_filter"] = "Cartoon Effect"
            if st.button("🌫️ Blur BG", use_container_width=True):
                st.session_state["selected_filter"] = "Background Blur"
            if st.button("🖤 B & W", use_container_width=True):
                st.session_state["selected_filter"] = "Grayscale"
            if st.button("☀️ Brighten", use_container_width=True):
                st.session_state["selected_filter"] = "Brightness"
        with qcol2:
            if st.button("✂️ Cutout", use_container_width=True):
                st.session_state["selected_filter"] = "Background Removal"
            if st.button("✏️ Sketch", use_container_width=True):
                st.session_state["selected_filter"] = "Canny Edge Detection"
            if st.button("📜 Vintage", use_container_width=True):
                st.session_state["selected_filter"] = "Sepia"
            if st.button("📷 Original", use_container_width=True):
                st.session_state["selected_filter"] = "Original"

        st.markdown("<hr style='margin: 12px 0; opacity: 0.2;'>", unsafe_allow_html=True)

        # Full Filter Dropdown for all 18
        current_index = (
            ALL_FILTER_NAMES.index(st.session_state["selected_filter"])
            if st.session_state["selected_filter"] in ALL_FILTER_NAMES
            else 0
        )
        selected_filter = st.selectbox(
            "📋 Or Choose From All 18 Filters:",
            ALL_FILTER_NAMES,
            index=current_index,
            key="filter_dropdown",
        )
        st.session_state["selected_filter"] = selected_filter

        # Dynamic Sliders with friendly descriptions
        st.markdown(f"#### ⚙️ Adjust: *{selected_filter}*")
        params = {}

        if selected_filter == "Cartoon Effect":
            params["num_bilateral"] = st.slider("Smoothness (Drawing feel)", 1, 6, 3, step=1)
            params["num_colors"] = st.slider("Color Levels (Anime palette)", 4, 32, 8, step=2)
            params["edge_kernel"] = st.slider("Outline Thickness", 3, 15, 7, step=2)

        elif selected_filter == "Background Blur":
            params["blur_strength"] = st.slider("Background Softness", 5, 75, 35, step=2)
            params["feather"] = st.slider("Edge Blending Smoothness", 1, 15, 5, step=1)
            params["threshold"] = st.slider("Person Detection Sensitivity", 0.1, 0.9, 0.4, step=0.05)
            params["use_ai"] = st.checkbox("AI High-Quality Mode", value=True)

        elif selected_filter == "Background Removal":
            params["background_type"] = st.selectbox(
                "Backdrop:",
                ["Transparent (PNG)", "White", "Black", "Studio Grey", "Custom Color"],
            )
            if params["background_type"] == "Custom Color":
                col = st.color_picker("Pick Backdrop Color", "#38BDF8")
                r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
                params["custom_color"] = (r, g, b)
            else:
                params["custom_color"] = (255, 255, 255)
            params["feather"] = st.slider("Edge Smoothing", 1, 15, 5, step=1)
            params["threshold"] = st.slider("Detection Sensitivity", 0.1, 0.9, 0.4, step=0.05)
            params["use_ai"] = st.checkbox("AI High-Quality Mode", value=True)

        elif selected_filter == "Sharpen":
            params["strength"] = st.slider("Detail Crispness", 0.1, 4.0, 1.5, step=0.1)

        elif selected_filter == "Brightness":
            params["brightness"] = st.slider("Light Level", -100, 100, 30, step=5)

        elif selected_filter == "Contrast":
            params["contrast"] = st.slider("Punch & Shadows", 0.1, 3.0, 1.5, step=0.1)

        elif selected_filter == "Saturation":
            params["saturation"] = st.slider("Color Vibrance", 0.0, 3.0, 1.5, step=0.1)

        elif selected_filter == "Sepia":
            params["intensity"] = st.slider("Vintage Warmth", 0.0, 1.0, 1.0, step=0.05)

        elif selected_filter == "Gaussian Blur":
            params["kernel_size"] = st.slider("Blur Amount", 3, 51, 15, step=2)
            params["sigma"] = st.slider("Blur Dispersion", 0.0, 10.0, 0.0, step=0.5)

        elif selected_filter == "Median Blur":
            params["kernel_size"] = st.slider("Filter Window Size", 3, 45, 11, step=2)

        elif selected_filter == "Canny Edge Detection":
            params["low_threshold"] = st.slider("Sensitivity (Fine Lines)", 0, 255, 50, step=5)
            params["high_threshold"] = st.slider("Main Contours", 0, 255, 150, step=5)

        elif selected_filter == "Edge Detection":
            params["method"] = st.selectbox("Style", ["Sobel", "Laplacian", "Prewitt"])
            params["kernel_size"] = st.slider("Line Width", 1, 7, 3, step=2)

        elif selected_filter == "Threshold":
            params["threshold_type"] = st.selectbox("Mode", ["Binary", "Binary Inverted", "Otsu"])
            if params["threshold_type"] != "Otsu":
                params["thresh"] = st.slider("Black / White Cutoff", 0, 255, 127, step=1)
            else:
                params["thresh"] = 0
            params["max_val"] = 255

        elif selected_filter == "Adaptive Threshold":
            params["method"] = st.selectbox("Method", ["Gaussian", "Mean"])
            params["block_size"] = st.slider("Window Size", 3, 51, 11, step=2)
            params["c_val"] = st.slider("Brightness Compensation", -20, 20, 2, step=1)
            params["max_val"] = 255

        elif selected_filter == "Emboss":
            params["strength"] = st.slider("3D Carve Depth", 0.2, 3.0, 1.0, step=0.1)

        else:
            st.caption("✨ Ready! No sliders needed for this filter.")

        st.markdown("<hr style='margin: 15px 0; opacity: 0.2;'>", unsafe_allow_html=True)
        st.markdown("#### 📐 Display Settings")
        comparison_view = st.selectbox(
            "Comparison View:",
            ["Side-by-Side (Original vs Filtered)", "Filtered Only", "Before / After Split (50/50)"],
            index=0,
        )

        max_dimension = 1080

    # Thread-safe sync for live stream
    LiveFilterHolder.set(selected_filter, params)

    return {
        "selected_filter": selected_filter,
        "params": params,
        "comparison_view": comparison_view,
        "max_dimension": max_dimension,
    }


def main():
    config = render_sidebar()
    selected_filter = config["selected_filter"]
    params = config["params"]
    comp_view = config["comparison_view"]
    max_dim = config["max_dimension"]

    # --- Header Title & Author ---
    st.markdown("<div class='main-title'>Real Time CV Filter Studio</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='lead-text'>Apply 18 real-time computer vision filters with live webcam streaming, AI background blur, and instant download. <b>Created by Bilal Butt</b>.</div>",
        unsafe_allow_html=True,
    )

    # Friendly Filter Info Banner
    info = FILTER_INFO.get(selected_filter, {"icon": "✨", "title": selected_filter, "desc": "Custom filter effect."})
    st.markdown(
        f"""
        <div class='filter-desc-box'>
            <b>{info['icon']} {info['title']}:</b> {info['desc']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Main Navigation Tabs (Super User-Friendly) ---
    tab_cam, tab_upload, tab_snapshot, tab_presets = st.tabs([
        "📹 Live Webcam Stream",
        "🖼️ Upload Your Photo",
        "📸 Quick Camera Snapshot",
        "🎨 Try Demo Photos",
    ])

    # 1. TAB: LIVE WEBCAM STREAM
    with tab_cam:
        st.markdown("#### 🔴 Live Browser Video Stream")
        st.markdown(
            """
            <div class='step-box'>
                <span class='step-pill'>👉 Step 1: Click <b>START</b> below & allow camera access</span>
                <span class='step-pill'>👉 Step 2: Pick any filter from the sidebar in real time!</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if HAS_WEBRTC:
            webrtc_streamer(
                key="cv-studio-live-stream",
                video_frame_callback=webrtc_video_frame_callback,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                async_processing=True,
            )
            st.markdown(
                f"""
                <div style="display: flex; gap: 12px; margin-top: 15px; max-width: 640px;">
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-val">{info['icon']} {selected_filter}</div>
                        <div class="metric-lbl">Active Live Filter</div>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-val">30+ FPS</div>
                        <div class="metric-lbl">Target Frame Rate</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("WebRTC package is initializing. You can also use 'Quick Camera Snapshot' in the next tab.")

    # 2. TAB: UPLOAD PHOTO
    with tab_upload:
        st.markdown("#### 📁 Upload an Image from Your Phone or Computer")
        uploaded_file = st.file_uploader(
            "Drop your photo here (JPG, PNG, WEBP):",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                img_data = utils.load_image(uploaded_file)
                render_image_processing_view(img_data, selected_filter, params, comp_view, max_dim)
            except Exception as e:
                st.error(f"Could not read image: {e}")
        else:
            st.info("💡 Click 'Browse files' above to choose a photo, or check the **'Try Demo Photos'** tab!")

    # 3. TAB: QUICK CAMERA SNAPSHOT
    with tab_snapshot:
        st.markdown("#### 📸 Take a Photo with Your Camera")
        cam_snap = st.camera_input("Smile and click 'Take Photo':", label_visibility="collapsed")
        if cam_snap is not None:
            try:
                img_data = utils.load_image(cam_snap)
                render_image_processing_view(img_data, selected_filter, params, comp_view, max_dim)
            except Exception as e:
                st.error(f"Error capturing snapshot: {e}")

    # 4. TAB: TRY DEMO PHOTOS
    with tab_presets:
        st.markdown("#### 🌟 Instant One-Click Demo Photos")
        presets = get_available_presets()
        if presets:
            col_p1, col_p2 = st.columns([2, 1])
            with col_p1:
                selected_preset = st.selectbox("Select Demo Picture:", list(presets.keys()))
            try:
                img_data = utils.load_image(presets[selected_preset])
                render_image_processing_view(img_data, selected_filter, params, comp_view, max_dim)
            except Exception as e:
                st.error(f"Error loading demo image: {e}")
        else:
            st.info("Presets directory empty.")


def render_image_processing_view(input_image: np.ndarray, selected_filter: str, params: dict, comp_view: str, max_dim: int):
    """Processes image and displays side-by-side comparison, metrics, and download button."""
    orig_h, orig_w = input_image.shape[:2]
    processed_input = utils.resize_image_max_dim(input_image, max_dim=max_dim)
    cur_h, cur_w = processed_input.shape[:2]

    # Process image
    processed_result, duration_ms, fps = process_image(processed_input, selected_filter, params)

    # Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{selected_filter}</div>
                <div class="metric-lbl">Active Filter</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{duration_ms:.1f} ms</div>
                <div class="metric-lbl">Processing Speed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{fps:.0f} FPS</div>
                <div class="metric-lbl">Throughput</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{cur_w} × {cur_h}</div>
                <div class="metric-lbl">Resolution</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    is_rgba = (processed_result.ndim == 3 and processed_result.shape[2] == 4)

    # Render Image View
    if comp_view == "Side-by-Side (Original vs Filtered)":
        c_orig, c_proc = st.columns(2)
        with c_orig:
            st.markdown("##### 📷 Original")
            st.image(processed_input, channels="RGB", use_container_width=True)
        with c_proc:
            st.markdown(f"##### ✨ Result ({selected_filter})")
            st.image(
                processed_result,
                channels="RGBA" if is_rgba else "RGB",
                use_container_width=True,
            )

    elif comp_view == "Before / After Split (50/50)":
        split_x = cur_w // 2
        split_view = processed_input.copy()
        if is_rgba:
            fg = processed_result[:, split_x:, :3]
            alpha = processed_result[:, split_x:, 3:4] / 255.0
            bg = processed_input[:, split_x:]
            split_view[:, split_x:] = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        else:
            split_view[:, split_x:] = processed_result[:, split_x:]

        cv2.line(split_view, (split_x, 0), (split_x, cur_h), (255, 255, 255), 3)
        cv2.putText(split_view, "BEFORE", (25, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(split_view, "AFTER", (split_x + 25, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        st.markdown(f"##### 🔀 Before (Left) vs {selected_filter} (Right)")
        st.image(split_view, channels="RGB", use_container_width=True)

    else:  # Filtered Only
        st.markdown(f"##### ✨ Result ({selected_filter})")
        st.image(
            processed_result,
            channels="RGBA" if is_rgba else "RGB",
            use_container_width=True,
        )

    # Big Friendly Download Button
    st.markdown("<br>", unsafe_allow_html=True)
    dcol1, dcol2 = st.columns([1, 2])
    with dcol1:
        if is_rgba:
            fmt, mime, ext = "PNG", "image/png", "png"
        else:
            fmt, mime, ext = "JPEG", "image/jpeg", "jpg"

        file_bytes = utils.convert_to_download_bytes(processed_result, format=fmt)
        clean_name = f"filter_{selected_filter.lower().replace(' ', '_')}.{ext}"

        st.download_button(
            label=f"⬇️ Download Photo ({fmt})",
            data=file_bytes,
            file_name=clean_name,
            mime=mime,
            use_container_width=True,
        )
    with dcol2:
        st.caption(f"✅ Ready! High-resolution {fmt} export ({len(file_bytes) // 1024} KB).")


if __name__ == "__main__":
    main()
