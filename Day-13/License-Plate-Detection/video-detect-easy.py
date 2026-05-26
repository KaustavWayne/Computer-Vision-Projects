import cv2
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

input_video = "sample-2.mp4"

output_video = "output_video.mp4"

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

    results = model(frame, verbose = False)[0]

    # ======================================================
    # DRAW DETECTIONS AUTOMATICALLY
    # ======================================================

    annotated_frame = results.plot()

    # ======================================================
    # SAVE FRAME
    # ======================================================

    out.write(annotated_frame)

    frame_count += 1

    print(f"Processed Frame: {frame_count}", end="\r")

# ==========================================================
# RELEASE RESOURCES
# ==========================================================

cap.release()

out.release()

cv2.destroyAllWindows()   # If your code does NOT use cv2.imshow(), you can skip this line

print("\n✅ Detection Completed!")

print(f"✅ Saved Video: {output_video}")