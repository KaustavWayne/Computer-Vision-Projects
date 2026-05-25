import cv2
from ultralytics import YOLO

# ==========================================================
# LOAD YOLO MODEL
# ==========================================================

best_model = YOLO("best-car.pt")

print("Classes:", best_model.names)

# ==========================================================
# START WEBCAM
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not opening")
    exit()

print("✅ Webcam started... Press Q to quit")

# ==========================================================
# WEBCAM LOOP
# ==========================================================

while True:

    # ======================================================
    # READ FRAME
    # ======================================================

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    # ======================================================
    # YOLO PREDICTION
    # ======================================================

    results = best_model.predict(
        frame,
        conf=0.4,
        imgsz=640,
        verbose=False
    )[0]

    # ======================================================
    # AUTO DRAW BOXES
    # ======================================================

    annotated_frame = results.plot()

    # ======================================================
    # SHOW FRAME
    # ======================================================

    cv2.imshow(
        "License Plate Detection - Webcam",
        annotated_frame
    )

    # ======================================================
    # PRESS Q TO QUIT
    # ======================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================================
# RELEASE WEBCAM
# ==========================================================

cap.release()

cv2.destroyAllWindows()

print("✅ Webcam closed")