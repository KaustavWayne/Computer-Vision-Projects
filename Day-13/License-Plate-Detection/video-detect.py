import cv2
import subprocess
from ultralytics import YOLO

# ==========================================================
# LOAD YOLO MODEL
# ==========================================================

model_path = "best-car.pt"

model = YOLO(model_path)

print("Classes:", model.names)

# ==========================================================
# VIDEO PATHS
# ==========================================================

input_video = "sample.mp4"

output_video = "output_video-ocr.mp4"

# ==========================================================
# OPEN VIDEO
# ==========================================================

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("❌ Video not opening")
    exit()

# ==========================================================
# VIDEO PROPERTIES
# ==========================================================

fps = int(cap.get(cv2.CAP_PROP_FPS))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}")
print(f"Width: {width}")
print(f"Height: {height}")

# ==========================================================
# VIDEO WRITER
# ==========================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

# ==========================================================
# PROCESS VIDEO
# ==========================================================

frame_count = 0

while cap.isOpened():

    # ======================================================
    # READ FRAME
    # ======================================================

    ret, frame = cap.read()

    if not ret:
        break

    # ======================================================
    # YOLO PREDICTION
    # ======================================================

    results = model.predict(
        frame,
        verbose=False
    )[0]

    # ======================================================
    # LOOP THROUGH DETECTED BOXES
    # ======================================================

    for box in results.boxes:

        # ==================================================
        # BOX COORDINATES
        # ==================================================

        x1, y1, x2, y2 = box.xyxy[0]

        x1, y1, x2, y2 = (
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        )

        # ==================================================
        # CONFIDENCE SCORE
        # ==================================================

        conf = float(box.conf[0])

        # ==================================================
        # CLASS ID
        # ==================================================

        cls = int(box.cls[0])

        # ==================================================
        # LABEL
        # ==================================================

        label = f"{model.names[cls]} {conf:.2f}"

        # ==================================================
        # DRAW RECTANGLE
        # ==================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ==================================================
        # DRAW TEXT
        # ==================================================

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ======================================================
    # SAVE FRAME
    # ======================================================

    out.write(frame)

    frame_count += 1

    print(f"Processed Frame: {frame_count}", end="\r")

# ==========================================================
# RELEASE RESOURCES
# ==========================================================

cap.release()

out.release()

print("\n✅ Detection Completed!")

print(f"✅ Saved Video: {output_video}")