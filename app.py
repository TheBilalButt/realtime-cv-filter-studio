"""
Real Time Computer Vision Filter Studio
Built by Bilal Butt

A high-performance, real-time Computer Vision application powered by
Streamlit, OpenCV, NumPy, and PyTorch LRASPP MobileNet.
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

# STUN servers for NAT/firewall traversal on Streamlit Cloud
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
    page_title="Real Time CV Filter Studio | Bilal Butt",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    /* Global styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #38bdf8;
        background-color: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 9999px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-lbl {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Streamlit widget tweaks */
    div[data-testid="stSidebarHeader"] {
        padding-top: 1rem;
    }
    .stDownloadButton button {
        background: linear-gradient(90deg, #0284c7 0%, #4f46e5 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s ease;
    }
    .stDownloadButton button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
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
    except Exception as e:
        return None


# Pre-warm model cache silently
_ = load_cached_segmentation_model()


def get_available_presets():
    """Returns sample preset images if available."""
    presets = {}
    sample_dir = os.path.join(os.path.dirname(__file__), "assets")
    if os.path.exists(os.path.join(sample_dir, "portrait.jpg")):
        presets["Sample Portrait (AI Demo)"] = os.path.join(sample_dir, "portrait.jpg")
    if os.path.exists(os.path.join(sample_dir, "landscape.jpg")):
        presets["Sample Landscape (Texture Demo)"] = os.path.join(sample_dir, "landscape.jpg")
    return presets


def render_sidebar():
    """Renders all controls in the sidebar and returns configuration dict."""
    with st.sidebar:
        st.markdown("### ⚙️ Studio Controls")
        st.markdown("<span class='badge'>Built by Bilal Butt</span>", unsafe_allow_html=True)

        # 1. Input Source Selection
        st.markdown("#### 1. Input Source")
        input_source = st.radio(
            "Select Media Input:",
            [
                "Live Camera (WebRTC Stream)",
                "Camera Snapshot",
                "Upload Image",
                "Preset Samples",
                "Local Desktop Camera (OpenCV)",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # 2. Filter Selection
        st.markdown("#### 2. Select Filter")
        filter_list = [
            "Original",
            "Grayscale",
            "Gaussian Blur",
            "Median Blur",
            "Sharpen",
            "Edge Detection",
            "Canny Edge Detection",
            "Threshold",
            "Adaptive Threshold",
            "Brightness",
            "Contrast",
            "Saturation",
            "Negative",
            "Sepia",
            "Emboss",
            "Cartoon Effect",
            "Background Blur",
            "Background Removal",
        ]
        selected_filter = st.selectbox(
            "Filter:",
            filter_list,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # 3. Dynamic Filter Parameters
        st.markdown(f"#### 3. Parameters: *{selected_filter}*")
        params = {}

        if selected_filter == "Gaussian Blur":
            params["kernel_size"] = st.slider("Kernel Size (Blur Radius)", 3, 51, 15, step=2)
            params["sigma"] = st.slider("Sigma (Dispersion)", 0.0, 10.0, 0.0, step=0.5)

        elif selected_filter == "Median Blur":
            params["kernel_size"] = st.slider("Kernel Size", 3, 45, 11, step=2)

        elif selected_filter == "Sharpen":
            params["strength"] = st.slider("Sharpening Strength", 0.1, 4.0, 1.5, step=0.1)

        elif selected_filter == "Edge Detection":
            params["method"] = st.selectbox("Algorithm", ["Sobel", "Laplacian", "Prewitt"])
            params["kernel_size"] = st.slider("Aperture / Kernel Size", 1, 7, 3, step=2)

        elif selected_filter == "Canny Edge Detection":
            params["low_threshold"] = st.slider("Low Hysteresis Threshold", 0, 255, 50, step=5)
            params["high_threshold"] = st.slider("High Hysteresis Threshold", 0, 255, 150, step=5)

        elif selected_filter == "Threshold":
            params["threshold_type"] = st.selectbox("Type", ["Binary", "Binary Inverted", "Otsu"])
            if params["threshold_type"] != "Otsu":
                params["thresh"] = st.slider("Cutoff Threshold", 0, 255, 127, step=1)
            else:
                params["thresh"] = 0
                st.info("Otsu automatically calculates the optimal threshold.")
            params["max_val"] = st.slider("Max Intensity", 50, 255, 255, step=5)

        elif selected_filter == "Adaptive Threshold":
            params["method"] = st.selectbox("Neighborhood Method", ["Gaussian", "Mean"])
            params["block_size"] = st.slider("Block Size (odd)", 3, 51, 11, step=2)
            params["c_val"] = st.slider("Constant C Subtracted", -20, 20, 2, step=1)
            params["max_val"] = 255

        elif selected_filter == "Brightness":
            params["brightness"] = st.slider("Brightness Offset", -100, 100, 30, step=5)

        elif selected_filter == "Contrast":
            params["contrast"] = st.slider("Contrast Factor", 0.1, 3.0, 1.5, step=0.1)

        elif selected_filter == "Saturation":
            params["saturation"] = st.slider("Saturation Multiplier", 0.0, 3.0, 1.5, step=0.1)

        elif selected_filter == "Sepia":
            params["intensity"] = st.slider("Tone Intensity", 0.0, 1.0, 1.0, step=0.05)

        elif selected_filter == "Emboss":
            params["strength"] = st.slider("Emboss Relief Depth", 0.2, 3.0, 1.0, step=0.1)

        elif selected_filter == "Cartoon Effect":
            params["num_bilateral"] = st.slider("Smoothing Passes", 1, 6, 3, step=1)
            params["num_colors"] = st.slider("Color Palette Granularity", 4, 32, 8, step=2)
            params["edge_kernel"] = st.slider("Edge Block Size", 3, 15, 7, step=2)

        elif selected_filter == "Background Blur":
            params["blur_strength"] = st.slider("Background Blur Strength", 5, 75, 35, step=2)
            params["feather"] = st.slider("Edge Feathering Radius", 1, 15, 5, step=1)
            params["threshold"] = st.slider("AI Confidence Threshold", 0.1, 0.9, 0.4, step=0.05)
            params["use_ai"] = st.checkbox("Use AI Segmentation (LRASPP)", value=True, help="Uncheck to use high-speed OpenCV fallback.")

        elif selected_filter == "Background Removal":
            params["background_type"] = st.selectbox(
                "Replacement Background:",
                ["Transparent (PNG)", "White", "Black", "Studio Grey", "Custom Color"],
            )
            if params["background_type"] == "Custom Color":
                col = st.color_picker("Choose Color", "#00FFCC")
                r = int(col[1:3], 16)
                g = int(col[3:5], 16)
                b = int(col[5:7], 16)
                params["custom_color"] = (r, g, b)
            else:
                params["custom_color"] = (255, 255, 255)
            params["feather"] = st.slider("Edge Feathering Radius", 1, 15, 5, step=1)
            params["threshold"] = st.slider("AI Confidence Threshold", 0.1, 0.9, 0.4, step=0.05)
            params["use_ai"] = st.checkbox("Use AI Segmentation (LRASPP)", value=True, help="Uncheck to use high-speed OpenCV fallback.")

        else:
            st.write("No adjustable parameters required for this filter.")

        st.markdown("---")

        # 4. Processing & Display Settings
        st.markdown("#### 4. Studio Settings")
        max_dimension = st.select_slider(
            "Max Processing Resolution (px):",
            options=[480, 720, 1080, 1440, 2160],
            value=1080,
            help="Images larger than this will be smoothly resized to guarantee high performance.",
        )

        comparison_view = st.selectbox(
            "Comparison Layout:",
            ["Side-by-Side (Original vs Processed)", "Processed Only", "Before / After Split"],
            index=0,
        )

    # Synchronize state for live WebRTC background thread
    LiveFilterHolder.set(selected_filter, params)

    return {
        "input_source": input_source,
        "selected_filter": selected_filter,
        "params": params,
        "max_dimension": max_dimension,
        "comparison_view": comparison_view,
    }


def process_image(image_rgb: np.ndarray, filter_name: str, params: dict):
    """
    Applies the specified filter and returns (processed_image, duration_ms, fps).
    """
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
    """Processes each frame received from the browser's webcam via WebRTC in real time."""
    img_bgr = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    cur_filter, cur_params = LiveFilterHolder.get()
    try:
        processed, _, _ = process_image(img_rgb, cur_filter, cur_params)
        if processed.ndim == 3 and processed.shape[2] == 4:
            alpha = processed[:, :, 3:4] / 255.0
            fg = processed[:, :, :3]
            bg = np.full_like(fg, 30)
            comp = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
            out_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
        else:
            out_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    except Exception:
        out_bgr = img_bgr

    return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")


def run_live_webcam_feed(selected_filter: str, params: dict, max_dim: int):
    """
    Executes a high-speed live webcam stream using OpenCV VideoCapture.
    Renders frames directly to an empty Streamlit placeholder with live FPS display.
    """
    st.info("💡 **Local Desktop OpenCV Mode Active.** Click **'Stop Webcam Feed'** below when finished.")

    col1, col2 = st.columns([3, 1])
    with col1:
        stop_button = st.button("⏹️ Stop Webcam Feed", type="primary", use_container_width=True)
    with col2:
        cam_id = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    cap = cv2.VideoCapture(int(cam_id))
    if not cap.isOpened():
        st.error(
            f"❌ Camera device index {cam_id} could not be opened by the server.\n\n"
            "👉 **On Streamlit Cloud**, cloud virtual machines have no physical webcam attached. "
            "Please switch to **'Live Camera (WebRTC Stream)'** in the sidebar to stream directly from your personal browser webcam!"
        )
        return

    # Optimize capture resolution for high responsiveness
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    perf_tracker = utils.PerformanceTracker(smoothing_window=15)

    try:
        while not stop_button:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("⚠️ Failed to capture frame from camera.")
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = utils.resize_image_max_dim(frame_rgb, max_dim=max_dim)

            t0 = time.perf_counter()
            processed, duration_ms, _ = process_image(frame_rgb, selected_filter, params)
            dur = time.perf_counter() - t0
            perf_tracker.record_duration(dur)

            # Display frame
            frame_placeholder.image(processed, channels="RGBA" if processed.ndim == 3 and processed.shape[2] == 4 else "RGB", use_container_width=True)

            # Update live stats
            stats_placeholder.markdown(
                f"""
                <div style="display: flex; gap: 15px; margin-top: 10px;">
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-val">{perf_tracker.current_fps:.1f}</div>
                        <div class="metric-lbl">Live FPS</div>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-val">{perf_tracker.avg_duration_ms:.1f} ms</div>
                        <div class="metric-lbl">Latency</div>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-val">{selected_filter}</div>
                        <div class="metric-lbl">Filter</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Brief pause to yield execution
            time.sleep(0.01)

    finally:
        cap.release()
        st.success("Webcam stream stopped.")


def main():
    config = render_sidebar()

    # --- Header Banner ---
    st.markdown("<div class='main-header'>Real Time Computer Vision Filter Studio</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>High-speed, responsive Computer Vision studio featuring 18 advanced image filters, AI background segmentation, and live webcam feed. <b>Built by Bilal Butt</b>.</div>",
        unsafe_allow_html=True,
    )

    input_source = config["input_source"]
    selected_filter = config["selected_filter"]
    params = config["params"]
    max_dim = config["max_dimension"]
    comp_view = config["comparison_view"]

    # 1. Handle Live WebRTC Browser Stream Mode
    if input_source == "Live Camera (WebRTC Stream)":
        st.markdown(f"### 📹 Real-Time Live Browser Camera ({selected_filter})")
        st.info(
            "💡 Click **'START'** below. Your browser will prompt to allow camera access. "
            "Your live video feed will stream and process in real time as you adjust filter parameters in the sidebar!"
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
                <div style="display: flex; gap: 15px; margin-top: 15px;">
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-val">{selected_filter}</div>
                        <div class="metric-lbl">Active Live Filter</div>
                    </div>
                    <div class="metric-card" style="flex: 2;">
                        <div class="metric-val">WebRTC Stream</div>
                        <div class="metric-lbl">Mode (Browser Client)</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ WebRTC module is loading. Please refresh the page or use 'Camera Snapshot' above.")
        return

    # 2. Handle Local Desktop OpenCV Camera Feed
    if input_source == "Local Desktop Camera (OpenCV)":
        run_live_webcam_feed(selected_filter, params, max_dim)
        return

    # 2. Acquire Image Data
    input_image = None

    if input_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file (JPG, PNG, WEBP, BMP):",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
        )
        if uploaded_file is not None:
            try:
                input_image = utils.load_image(uploaded_file)
            except Exception as e:
                st.error(f"Error loading uploaded image: {e}")
        else:
            st.info("👆 Upload an image from your computer to begin, or switch to 'Preset Samples' in the sidebar.")

    elif input_source == "Camera Snapshot":
        camera_shot = st.camera_input("Take a photo with your webcam:")
        if camera_shot is not None:
            try:
                input_image = utils.load_image(camera_shot)
            except Exception as e:
                st.error(f"Error reading webcam photo: {e}")

    elif input_source == "Preset Samples":
        presets = get_available_presets()
        if presets:
            selected_preset_name = st.selectbox("Select Preset Demo:", list(presets.keys()))
            try:
                input_image = utils.load_image(presets[selected_preset_name])
            except Exception as e:
                st.error(f"Error loading preset: {e}")
        else:
            st.warning("No preset sample files found in assets directory.")

    # 3. Process and Display
    if input_image is not None:
        # Resize if exceeding max dimension for guaranteed responsiveness
        orig_h, orig_w = input_image.shape[:2]
        processed_input = utils.resize_image_max_dim(input_image, max_dim=max_dim)
        cur_h, cur_w = processed_input.shape[:2]

        # Execute filter
        try:
            processed_result, duration_ms, fps = process_image(
                processed_input, selected_filter, params
            )
        except Exception as e:
            st.error(f"Error applying filter '{selected_filter}': {e}")
            return

        # Render Performance Metrics Bar
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">{selected_filter}</div>
                    <div class="metric-lbl">Active Filter</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">{duration_ms:.1f} ms</div>
                    <div class="metric-lbl">Processing Latency</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">{fps:.1f}</div>
                    <div class="metric-lbl">Effective FPS</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">{cur_w} × {cur_h}</div>
                    <div class="metric-lbl">Resolution ({"Resized" if (cur_w != orig_w or cur_h != orig_h) else "Original"})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Render Image Display based on Comparison Layout
        is_rgba = (processed_result.ndim == 3 and processed_result.shape[2] == 4)

        if comp_view == "Side-by-Side (Original vs Processed)":
            col_orig, col_proc = st.columns(2)
            with col_orig:
                st.markdown("##### 📷 Original Image")
                st.image(processed_input, channels="RGB", use_container_width=True)
            with col_proc:
                st.markdown(f"##### ✨ Processed Result ({selected_filter})")
                st.image(
                    processed_result,
                    channels="RGBA" if is_rgba else "RGB",
                    use_container_width=True,
                )

        elif comp_view == "Before / After Split":
            # 50/50 horizontal split visualization
            split_x = cur_w // 2
            split_view = processed_input.copy()
            if is_rgba:
                # Merge RGB with RGBA foreground
                fg = processed_result[:, split_x:, :3]
                alpha = processed_result[:, split_x:, 3:4] / 255.0
                bg = processed_input[:, split_x:]
                split_view[:, split_x:] = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
            else:
                split_view[:, split_x:] = processed_result[:, split_x:]

            # Draw divider line
            cv2.line(split_view, (split_x, 0), (split_x, cur_h), (255, 255, 255), 2)
            cv2.putText(split_view, "BEFORE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(split_view, "AFTER", (split_x + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            st.markdown(f"##### 🔀 Split View: Original (Left) vs {selected_filter} (Right)")
            st.image(split_view, channels="RGB", use_container_width=True)

        else:  # Processed Only
            st.markdown(f"##### ✨ Processed Result ({selected_filter})")
            st.image(
                processed_result,
                channels="RGBA" if is_rgba else "RGB",
                use_container_width=True,
            )

        # Download Button Section
        st.markdown("<br>", unsafe_allow_html=True)
        d_col1, d_col2 = st.columns([1, 3])
        with d_col1:
            if is_rgba:
                out_format = "PNG"
                mime = "image/png"
                ext = "png"
            else:
                out_format = "JPEG"
                mime = "image/jpeg"
                ext = "jpg"

            download_bytes = utils.convert_to_download_bytes(processed_result, format=out_format)
            clean_filename = f"filtered_{selected_filter.lower().replace(' ', '_')}.{ext}"

            st.download_button(
                label=f"⬇️ Download Image ({out_format})",
                data=download_bytes,
                file_name=clean_filename,
                mime=mime,
                use_container_width=True,
            )
        with d_col2:
            st.caption(f"Ready for export in {out_format} format ({len(download_bytes) // 1024} KB).")


if __name__ == "__main__":
    main()
