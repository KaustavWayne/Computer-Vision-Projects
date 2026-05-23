import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import time
import io
from pathlib import Path

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PotHole Vision AI",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CSS  (dark neon theme + animations)
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

    /* ── Root & background ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
        color: #e0e0ff;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060612 0%, #0d1b2a 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stHeader"] { background: transparent; }

    /* ── Animated neon title ── */
    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.2rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00f5ff, #bf5fff, #ff6b6b, #00f5ff);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 4s ease infinite;
        text-shadow: 0 0 40px rgba(0,245,255,0.3);
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #7fc8f8;
        font-size: 1.05rem;
        letter-spacing: 0.15em;
        margin-bottom: 1.5rem;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ── Glowing cards ── */
    .glass-card {
        background: rgba(13, 27, 42, 0.8);
        border: 1px solid rgba(0, 245, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.08);
        backdrop-filter: blur(12px);
        transition: box-shadow 0.3s;
    }
    .glass-card:hover { box-shadow: 0 0 35px rgba(0, 245, 255, 0.2); }

    /* ── Stat badges ── */
    .stat-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        margin: 0.2rem;
    }
    .badge-pothole  { background: rgba(255,107,107,0.2); border:1px solid #ff6b6b; color:#ff6b6b; }
    .badge-drain    { background: rgba(0,245,255,0.2);   border:1px solid #00f5ff; color:#00f5ff; }
    .badge-sewer    { background: rgba(191,95,255,0.2);  border:1px solid #bf5fff; color:#bf5fff; }
    .badge-total    { background: rgba(255,209,0,0.2);   border:1px solid #ffd100; color:#ffd100; }

    /* ── Pulse animation for LIVE badge ── */
    .live-badge {
        display: inline-flex; align-items: center; gap: 0.5rem;
        background: rgba(255,59,59,0.15); border:1px solid #ff3b3b;
        border-radius:20px; padding:0.3rem 0.9rem;
        color:#ff3b3b; font-weight:700; font-size:0.85rem;
        animation: pulse 1.5s infinite;
    }
    .live-dot { width:8px; height:8px; border-radius:50%; background:#ff3b3b; 
                animation: blink 1s step-end infinite; }
    @keyframes pulse { 0%,100%{box-shadow:0 0 6px rgba(255,59,59,0.4);} 50%{box-shadow:0 0 18px rgba(255,59,59,0.8);} }
    @keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0;} }

    /* ── Sidebar nav pills ── */
    .nav-pill {
        display:block; padding:0.7rem 1rem; border-radius:10px;
        margin-bottom:0.4rem; cursor:pointer; transition:all 0.25s;
        border:1px solid transparent; font-weight:500;
    }
    .nav-pill:hover { border-color:rgba(0,245,255,0.4); background:rgba(0,245,255,0.07); }

    /* ── Progress bar override ── */
    [data-testid="stProgress"] > div > div { background: linear-gradient(90deg,#00f5ff,#bf5fff) !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #00f5ff22, #bf5fff22);
        border: 1px solid #00f5ff66;
        color: #e0e0ff;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00f5ff44, #bf5fff44);
        border-color: #00f5ff;
        box-shadow: 0 0 15px rgba(0,245,255,0.3);
        transform: translateY(-1px);
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(0,245,255,0.35) !important;
        border-radius: 14px !important;
        background: rgba(0,245,255,0.04) !important;
        transition: border-color 0.3s;
    }
    [data-testid="stFileUploader"]:hover { border-color: rgba(0,245,255,0.7) !important; }

    /* ── Sidebar radio + label fix ── */
    [data-testid="stRadio"] label { color: #e0e0ff !important; font-size:0.95rem !important; }
    [data-testid="stRadio"] { background: rgba(0,245,255,0.04); border-radius:10px; padding:0.5rem; }
    [data-testid="stSidebarContent"] * { color: #e0e0ff; }
    .stSlider label { color: #7fc8f8 !important; }
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] { color: #3a5a7a !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width:6px; }
    ::-webkit-scrollbar-track { background:#0a0a1a; }
    ::-webkit-scrollbar-thumb { background:#1e3a5f; border-radius:3px; }
    ::-webkit-scrollbar-thumb:hover { background:#00f5ff44; }

    /* ── Divider ── */
    hr { border-color: rgba(0,245,255,0.15) !important; }

    /* ── Metric cards: label + value + delta ── */
    [data-testid="stMetric"] {
        background: rgba(0, 245, 255, 0.05);
        border: 1px solid rgba(0, 245, 255, 0.15);
        border-radius: 12px;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] {
        color: #7fc8f8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        font-family: 'Orbitron', sans-serif !important;
    }
    [data-testid="stMetricDelta"] { color: #00f5ff !important; }

    /* ── Dataframe / table dark theme ── */
    [data-testid="stDataFrame"] iframe,
    .stDataFrame iframe { background: #0d1b2a !important; }
    [data-testid="stDataFrame"] > div {
        background: #0d1b2a !important;
        border: 1px solid rgba(0,245,255,0.2) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    /* Override default white table cells */
    .dvn-scroller, .stDataFrameGlideDataEditor { background: #0d1b2a !important; }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, rgba(0,245,255,0.15), rgba(191,95,255,0.15)) !important;
        border: 1px solid #00f5ff88 !important;
        color: #e0e0ff !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        width: 100%;
        transition: all 0.3s;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(135deg, rgba(0,245,255,0.3), rgba(191,95,255,0.3)) !important;
        border-color: #00f5ff !important;
        box-shadow: 0 0 18px rgba(0,245,255,0.35) !important;
        transform: translateY(-1px);
        color: #ffffff !important;
    }

    /* ── Captions, markdown text ── */
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: #7fc8f8 !important;
        font-size: 0.85rem !important;
    }
    p, .stMarkdown p { color: #c8d8f0 !important; }

    /* ── Success / error / warning boxes ── */
    [data-testid="stAlert"] { border-radius: 10px !important; }

    /* ── Spinner text ── */
    .stSpinner > div { color: #00f5ff !important; }

    /* ── Number input ── */
    [data-testid="stNumberInput"] label { color: #7fc8f8 !important; }
    [data-testid="stNumberInput"] input { 
        background: #0d1b2a !important; 
        color: #e0e0ff !important; 
        border: 1px solid rgba(0,245,255,0.3) !important;
        border-radius: 8px !important;
    }

    /* ── Confidence meter ── */
    .conf-bar-wrap { width:100%; background:#111827; border-radius:99px; height:8px; margin-top:4px; }
    .conf-bar { height:8px; border-radius:99px; background:linear-gradient(90deg,#00f5ff,#bf5fff); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
MODEL_PATH  = Path(__file__).parent / "best.pt"
CLASS_NAMES = ["Drain Hole", "Pothole", "Sewer Cover"]
CLASS_COLORS = {
    "Drain Hole":   (0, 245, 255),    # cyan
    "Pothole":      (255, 107, 107),  # red-pink
    "Sewer Cover":  (191, 95, 255),   # purple
}
BADGE_CSS = {
    "Drain Hole":  "badge-drain",
    "Pothole":     "badge-pothole",
    "Sewer Cover": "badge-sewer",
}

# ──────────────────────────────────────────────
# MODEL LOADER  (cached across reruns)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(str(MODEL_PATH))

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def get_counts(results) -> dict:
    """Extract per-class detection counts from YOLO results."""
    counts = {c: 0 for c in CLASS_NAMES}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        counts[label] = counts.get(label, 0) + 1
    return counts


def counts_html(counts: dict) -> str:
    total = sum(counts.values())
    parts = [f'<span class="stat-badge badge-total">🔍 Total: {total}</span>']
    for cls, cnt in counts.items():
        if cnt:
            parts.append(f'<span class="stat-badge {BADGE_CSS[cls]}">{cls}: {cnt}</span>')
    return " ".join(parts)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-family:\'Orbitron\',sans-serif; font-size:1.3rem; '
        'font-weight:700; color:#00f5ff; margin-bottom:0.2rem;">🚧 PotHole Vision</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<p style="color:#7fc8f8; font-size:0.8rem;">YOLOv8 Detection Suite</p>', unsafe_allow_html=True)
    st.markdown("---")

    mode = st.radio(
        "**Detection Mode**",
        ["📸  Image", "🎥  Video", "📡  Live Webcam"],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown("**⚙️ Model Settings**")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.95, 0.40, 0.05)
    iou_thresh  = st.slider("IoU Threshold (NMS)",  0.10, 0.95, 0.45, 0.05)

    st.markdown("---")
    st.markdown("**🏷️ Detectable Classes**")
    for cls, color in CLASS_COLORS.items():
        r, g, b = color
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">'
            f'<div style="width:14px;height:14px;border-radius:4px;background:rgb({r},{g},{b});flex-shrink:0;"></div>'
            f'<span style="color:#e0e0ff;font-size:0.9rem;">{cls}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<p style="color:#3a5a7a; font-size:0.75rem; text-align:center;">'
        'Model: <code style="color:#00f5ff;">best.pt</code><br>Backbone: YOLOv8</p>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🚧 PotHole Vision AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">REAL-TIME ROAD HAZARD DETECTION  ·  YOLOV8  ·  DRAIN · POTHOLE · SEWER</p>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
model_status = st.empty()
with model_status.container():
    with st.spinner("🔄 Loading YOLOv8 model — please wait…"):
        try:
            model = load_model()
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()
model_status.success("✅  Model **best.pt** loaded successfully — YOLOv8 ready!", icon="🤖")

st.markdown("---")

# ══════════════════════════════════════════════
# ① IMAGE MODE
# ══════════════════════════════════════════════
if mode == "📸  Image":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📸 Image Detection")
    st.caption("Upload JPG / PNG / WEBP images for instant pothole analysis.")

    uploaded_files = st.file_uploader(
        "Drop images here",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files:
        for uploaded in uploaded_files:
            img_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr   = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            col_orig, col_det = st.columns(2, gap="medium")

            with col_orig:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**🖼️ Original**")
                st.image(img_rgb, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with st.spinner(f"Analyzing **{uploaded.name}**…"):
                t0      = time.time()
                results = model.predict(img_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)
                elapsed = time.time() - t0

            # ── YOLO native plot ──
            annotated_bgr = results[0].plot()          # BGR ndarray
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            counts        = get_counts(results)

            with col_det:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**🔍 Detections**")
                st.image(annotated_rgb, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Stats row
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            total = sum(counts.values())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⏱️ Inference", f"{elapsed*1000:.0f} ms")
            c2.metric("🔴 Potholes",  counts.get("Pothole", 0))
            c3.metric("🔵 Drain Holes", counts.get("Drain Hole", 0))
            c4.metric("🟣 Sewer Covers", counts.get("Sewer Cover", 0))
            st.markdown(counts_html(counts), unsafe_allow_html=True)

            # Per-detection confidence table (custom HTML — works with dark theme)
            boxes = results[0].boxes
            if len(boxes):
                st.markdown("**📋 Detection Details**")
                badge_color = {"Pothole": "#ff6b6b", "Drain Hole": "#00f5ff", "Sewer Cover": "#bf5fff"}
                rows_html = ""
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < conf_thresh:
                        continue
                    cid  = int(box.cls[0])
                    lbl  = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    col  = badge_color.get(lbl, "#e0e0ff")
                    bar_w = int(conf * 100)
                    rows_html += (
                        f'<tr>'
                        f'<td><span style="background:{col}22;border:1px solid {col};color:{col};'
                        f'padding:3px 10px;border-radius:20px;font-weight:600;font-size:0.82rem;">{lbl}</span></td>'
                        f'<td style="color:#e0e0ff;">'
                        f'  <div style="display:flex;align-items:center;gap:8px;">'
                        f'    <div style="flex:1;background:#1a2a3a;border-radius:99px;height:7px;">'
                        f'      <div style="width:{bar_w}%;height:7px;border-radius:99px;'
                        f'background:linear-gradient(90deg,#00f5ff,#bf5fff);"></div></div>'
                        f'    <span style="min-width:42px;color:#fff;font-weight:600;">{conf:.1%}</span>'
                        f'  </div></td>'
                        f'<td style="color:#7fc8f8;font-family:monospace;font-size:0.82rem;">{x1},{y1} → {x2},{y2}</td>'
                        f'</tr>'
                    )
                if rows_html:
                    st.markdown(
                        f'<table style="width:100%;border-collapse:collapse;">'
                        f'<thead><tr>'
                        f'<th style="text-align:left;color:#00f5ff;padding:8px 12px;'
                        f'border-bottom:1px solid rgba(0,245,255,0.2);font-size:0.82rem;letter-spacing:0.08em;">CLASS</th>'
                        f'<th style="text-align:left;color:#00f5ff;padding:8px 12px;'
                        f'border-bottom:1px solid rgba(0,245,255,0.2);font-size:0.82rem;letter-spacing:0.08em;">CONFIDENCE</th>'
                        f'<th style="text-align:left;color:#00f5ff;padding:8px 12px;'
                        f'border-bottom:1px solid rgba(0,245,255,0.2);font-size:0.82rem;letter-spacing:0.08em;">BOUNDING BOX</th>'
                        f'</tr></thead>'
                        f'<tbody style="background:rgba(0,0,0,0);">{rows_html}</tbody>'
                        f'</table>',
                        unsafe_allow_html=True,
                    )

            st.markdown("</div>", unsafe_allow_html=True)

            # Download annotated image
            buf = io.BytesIO()
            Image.fromarray(annotated_rgb).save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Annotated Image",
                data=buf.getvalue(),
                file_name=f"detected_{uploaded.name}",
                mime="image/png",
            )
            st.markdown("---")

# ══════════════════════════════════════════════
# ② VIDEO MODE
# ══════════════════════════════════════════════
elif mode == "🎥  Video":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎥 Video Detection")
    st.caption("Upload MP4 / AVI / MOV video files. Frames are processed and re-assembled.")

    uploaded_video = st.file_uploader("Drop video here", type=["mp4", "avi", "mov", "mkv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"📹 **{uploaded_video.name}** · {total_frames} frames · {fps:.1f} FPS · {w}×{h}")

        frame_skip = st.slider("Process every N-th frame (speed vs quality)", 1, 10, 2)
        process_btn = st.button("🚀 Start Detection", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if process_btn:
            out_path = tmp_path.replace(".mp4", "_out.mp4")
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            out      = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            progress_bar  = st.progress(0)
            status_text   = st.empty()
            preview_slot  = st.empty()
            count_slot    = st.empty()

            agg_counts = {c: 0 for c in CLASS_NAMES}
            frame_idx  = 0
            last_frame = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if frame_idx % frame_skip == 0:
                    results    = model.predict(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
                    ann        = results[0].plot()      # BGR ndarray
                    cnt        = get_counts(results)
                    for k, v in cnt.items():
                        agg_counts[k] += v
                    last_frame = ann
                else:
                    ann = last_frame if last_frame is not None else frame

                out.write(ann)

                pct = frame_idx / total_frames
                progress_bar.progress(min(pct, 1.0))
                status_text.markdown(
                    f'<p style="color:#7fc8f8;">Processing frame {frame_idx}/{total_frames} '
                    f'({pct:.0%})</p>', unsafe_allow_html=True
                )

                # Live preview every 15 frames
                if frame_idx % 15 == 0 and last_frame is not None:
                    preview_slot.image(
                        cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_idx}",
                        use_container_width=True,
                    )
                    count_slot.markdown(counts_html(agg_counts), unsafe_allow_html=True)

            cap.release()
            out.release()
            os.unlink(tmp_path)

            progress_bar.progress(1.0)
            status_text.success("✅ Video processing complete!")

            # Summary
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Aggregated Detection Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Total Potholes",    agg_counts.get("Pothole", 0))
            c2.metric("🔵 Drain Holes",        agg_counts.get("Drain Hole", 0))
            c3.metric("🟣 Sewer Covers",       agg_counts.get("Sewer Cover", 0))
            st.markdown(counts_html(agg_counts), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Download
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Processed Video",
                    data=f.read(),
                    file_name=f"detected_{uploaded_video.name}",
                    mime="video/mp4",
                )
            os.unlink(out_path)

# ══════════════════════════════════════════════
# ③ LIVE WEBCAM MODE
# ══════════════════════════════════════════════
elif mode == "📡  Live Webcam":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📡 Live Webcam Detection")
    st.markdown(
        '<span class="live-badge"><span class="live-dot"></span>LIVE</span>',
        unsafe_allow_html=True,
    )
    st.caption("Real-time detection from your webcam.")

    # ── Tips panel ──
    st.markdown(
        '<div style="background:rgba(0,245,255,0.06);border:1px solid rgba(0,245,255,0.2);'
        'border-radius:10px;padding:0.8rem 1rem;margin:0.5rem 0;">'
        '<p style="color:#00f5ff;font-weight:700;margin:0 0 0.4rem;">💡 Tips for best results</p>'
        '<ul style="color:#c8d8f0;margin:0;padding-left:1.2rem;font-size:0.88rem;">'
        '<li>Hold the object <b>30–60 cm</b> from the camera and keep it <b>steady</b> for 1–2 s</li>'
        '<li>Use <b>good lighting</b> — avoid pointing the camera at bright screens or dark areas</li>'
        '<li>Lower the <b>Confidence Threshold</b> in the sidebar (try <b>0.20–0.30</b>) for harder shots</li>'
        '<li>Use <b>Resize Input</b> below to boost FPS if detection feels laggy</li>'
        '</ul></div>',
        unsafe_allow_html=True,
    )

    # ── Controls ──
    col_a, col_b = st.columns(2)
    cam_index  = col_a.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
    max_frames = col_b.slider("Max frames to capture", 30, 500, 200, 10)

    col_c, col_d = st.columns(2)
    resize_w   = col_c.selectbox(
        "Resize input width (smaller = faster FPS)",
        [640, 480, 320],
        index=1,
        help="Resizes the frame fed to YOLO — smaller = faster inference, larger = more detail",
    )
    warmup_n   = col_d.number_input(
        "Warmup frames (autofocus)", min_value=0, max_value=60, value=25, step=5,
        help="Discards the first N frames so your camera autofocus settles before detection starts",
    )

    start_btn = st.button("▶️ Start Webcam", use_container_width=True)
    
    # ── Persistent Summary Slot (Always visible if previous session exists) ──
    summary_slot = st.empty()
    
    if "webcam_results" in st.session_state:
        res = st.session_state.webcam_results
        with summary_slot.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.success(f"📊 Previous Session Summary ({res['frames']} frames)")
            st.markdown(counts_html(res['counts']), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Potholes",    res['counts'].get("Pothole", 0))
            c2.metric("🔵 Drain Holes", res['counts'].get("Drain Hole", 0))
            c3.metric("🟣 Sewer Covers", res['counts'].get("Sewer Cover", 0))
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if start_btn:
        summary_slot.empty()
        cap = cv2.VideoCapture(int(cam_index))

        # Basic resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            st.error(f"❌ Cannot open camera {cam_index}.", icon="🚫")
        else:
            if warmup_n > 0:
                warmup_bar = st.progress(0, text="🔆 Warming up...")
                for wi in range(int(warmup_n)):
                    cap.read()
                    warmup_bar.progress((wi + 1) / int(warmup_n))
                warmup_bar.empty()

            metrics_slot = st.empty()
            frame_slot   = st.empty()
            count_slot   = st.empty()
            
            # Use session state for stop button to make it more responsive
            if 'stop_clicked' not in st.session_state:
                st.session_state.stop_clicked = False
            
            st.button("⏹️ Emergency Stop", on_click=lambda: st.session_state.update({"stop_clicked": True}))

            agg_counts = {c: 0 for c in CLASS_NAMES}
            frame_num  = 0
            fps_ring   = []

            while frame_num < max_frames and not st.session_state.stop_clicked:
                ret, frame = cap.read()
                if not ret: break

                # Faster resizing
                if resize_w < frame.shape[1]:
                    scale = resize_w / frame.shape[1]
                    frame = cv2.resize(frame, (resize_w, int(frame.shape[0] * scale)))

                t0 = time.time()
                # Use imgsz=320 for speed if resize_w is small
                results = model.predict(frame, conf=conf_thresh, iou=iou_thresh, imgsz=320 if resize_w <= 320 else 640, verbose=False)
                elapsed = time.time() - t0
                fps_ring.append(elapsed)
                if len(fps_ring) > 8: fps_ring.pop(0)
                fps_now = 1.0 / (sum(fps_ring) / len(fps_ring)) if fps_ring else 0

                ann = results[0].plot() # BGR
                cnt = get_counts(results)
                for k, v in cnt.items(): agg_counts[k] += v

                # Convert to RGB only once for Display
                ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                
                # Overlay info
                cv2.putText(ann_rgb, f"{fps_now:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 245, 255), 2)

                # Update live slots
                with metrics_slot.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("⚡ FPS", f"{fps_now:.1f}")
                    m2.metric("⏱️ Inf", f"{elapsed*1000:.0f}ms")
                    m3.metric("🔍 Det", sum(cnt.values()))

                frame_slot.image(ann_rgb, channels="RGB", use_container_width=True)
                count_slot.markdown(counts_html(cnt), unsafe_allow_html=True)

                frame_num += 1

            cap.release()
            st.session_state.stop_clicked = False # reset for next time
            
            # Final Save for persistence
            st.session_state.webcam_results = {"counts": agg_counts, "frames": frame_num}
            st.rerun()



# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; padding: 1rem 0 0.5rem;">'
    '<p style="color:#7fc8f8; font-size:1rem; font-weight:600; margin-bottom:0.2rem;">'
    'Built with <span style="color:#ff6b6b;">♥</span> by '
    '<span style="background:linear-gradient(90deg,#00f5ff,#bf5fff);-webkit-background-clip:text;'
    '-webkit-text-fill-color:transparent;font-weight:700;font-family:\'Orbitron\',sans-serif;'
    'font-size:1rem;">Kaustav Roy Chowdhury</span></p>'
    '<p style="color:#3a5a7a; font-size:0.78rem; margin-top:0.1rem;">'
    '🚧 PotHole Vision AI &nbsp;·&nbsp; YOLOv8 &nbsp;·&nbsp; '
    'Drain Hole &nbsp;|&nbsp; Pothole &nbsp;|&nbsp; Sewer Cover</p>'
    '</div>',
    unsafe_allow_html=True,
)