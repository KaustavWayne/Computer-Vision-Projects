import streamlit as st
import cv2
import os
import json

from src.detector import PlateDetector
from src.metadata import generate_metadata
from src.utils import get_images_from_dir

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Car Number Plate Detection",
    page_icon="🚗",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "ready" not in st.session_state:
    st.session_state.ready = False

if "metadata" not in st.session_state:
    st.session_state.metadata = None

if "search_clicked" not in st.session_state:
    st.session_state.search_clicked = False

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
body { background-color: #0e1117; }
.card {
    background: #161b22;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.caption {
    text-align: center;
    color: #c9d1d9;
    font-size: 15px;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    padding: 8px 18px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🚗 Car Number Plate Detection")
st.markdown(
    "<p style='color:#c9d1d9;font-size:16px;'>YOLO-powered detection & search</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# MODE
# --------------------------------------------------
mode = st.radio(
    "Choose an option:",
    ["Process new images", "Load existing metadata"]
)

# ==================================================
# PROCESS NEW IMAGES
# ==================================================
if mode == "Process new images":

    source = st.radio(
        "Image source:",
        ["Load from directory path", "Upload new images"]
    )

    # -------- DIRECTORY --------
    if source == "Load from directory path":
        dir_path = st.text_input(
            "Image directory path", placeholder=r"D:\CAMPUSX\Computer Vision Projects\Car_Plate_Search\data\raw")


        if st.button("Run Inference"):
            if not dir_path.strip():
                st.error("Please provide directory path")
            else:
                images = get_images_from_dir(dir_path)

                if not images:
                    st.error("No images found in this directory")
                else:
                    detector = PlateDetector("models/best.pt")
                    metadata = generate_metadata(images, detector)

                    st.session_state.metadata = metadata
                    st.session_state.ready = True
                    st.session_state.search_clicked = False

                    st.success(
                        f"Successfully processed {len(metadata)} images"
                    )

    # -------- UPLOAD --------
    if source == "Upload new images":
        files = st.file_uploader(
            "Upload images",
            type=["jpg","jpeg","png"],
            accept_multiple_files=True
        )

        if st.button("Run Inference"):
            if not files:
                st.error("Please upload at least one image")
            else:
                paths = []
                for f in files:
                    p = os.path.join(UPLOAD_DIR, f.name)
                    with open(p, "wb") as out:
                        out.write(f.getbuffer())
                    paths.append(p)

                detector = PlateDetector("models/best.pt")
                metadata = generate_metadata(paths, detector)

                st.session_state.metadata = metadata
                st.session_state.ready = True
                st.session_state.search_clicked = False

                st.success(
                    f"Successfully processed {len(metadata)} images"
                )

# ==================================================
# LOAD EXISTING METADATA
# ==================================================
if mode == "Load existing metadata":
    meta_file = st.file_uploader("Upload metadata.json", type=["json"])

    if st.button("Load Metadata"):
        if meta_file is None:
            st.error("Please upload metadata.json")
        else:
            st.session_state.metadata = json.load(meta_file)
            st.session_state.ready = True
            st.session_state.search_clicked = False

            st.success(
                f"Successfully loaded metadata for {len(st.session_state.metadata)} images"
            )

# ==================================================
# SEARCH BUTTON
# ==================================================
if st.session_state.ready:
    if st.button("Search Images"):
        st.session_state.search_clicked = True

# ==================================================
# RESULTS (ONLY AFTER SEARCH)
# ==================================================
if (
    st.session_state.ready
    and st.session_state.metadata
    and st.session_state.search_clicked
):

    st.subheader("🔎 Results")

    show_boxes = st.checkbox("Show bounding boxes", True)
    grid_cols = st.slider("Grid columns", 1, 4, 2)

    results = [
        (img, info)
        for img, info in st.session_state.metadata.items()
        if info["plate_count"] > 0
    ]

    st.markdown(f"### Results: {len(results)} matching images")

    detector = PlateDetector("models/best.pt")
    cols = st.columns(grid_cols)

    for i, (img_path, info) in enumerate(results):
        img = cv2.imread(img_path)

        if show_boxes:
            img = detector.draw_boxes(img_path, info["detections"])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with cols[i % grid_cols]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown(
                f"<div class='caption'>Plates detected: {info['plate_count']}</div>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# RESET
# ==================================================
if st.button("Reset Application"):
    st.session_state.ready = False
    st.session_state.metadata = None
    st.session_state.search_clicked = False
    st.experimental_rerun()
