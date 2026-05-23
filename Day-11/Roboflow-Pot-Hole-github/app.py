import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# =========================
# LOAD MODEL
# =========================
model = YOLO("best.pt")

st.title("🚧 Pothole Detection System")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # YOLO Detection
    results = model(image, conf=0.4, imgsz=640, verbose=False)

    # Plot detections
    annotated_image = results[0].plot()

    st.image(
        annotated_image,
        caption="Detected Potholes",
        use_container_width=True
    )