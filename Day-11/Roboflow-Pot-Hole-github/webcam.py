import cv2
from ultralytics import YOLO

# ==============================
# LOAD TRAINED YOLO MODEL
# ==============================
model = YOLO("best.pt")

# ==============================
# START WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not opening")
    exit()

print("✅ Webcam started... Press Q to quit")

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    # ==============================
    # YOLO DETECTION
    # ==============================
    results = model(frame, conf=0.4, imgsz=640, verbose=False)
    # results = model.predict(frame, conf=0.4, imgsz=640, verbose=False)[0]

    # Get first result
    annotated_frame = results[0].plot()

    # ==============================
    # SHOW OUTPUT
    # ==============================
    cv2.imshow("Pothole Detection", annotated_frame)

    # Quit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# RELEASE
# ==============================
cap.release()
cv2.destroyAllWindows()

print("✅ Webcam closed")