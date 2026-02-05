import cv2
from ultralytics import YOLO

# -------------------------------
# Load YOLO model
# -------------------------------
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

print("YOLO Number Plate model loaded")

# -------------------------------
# Start Webcam
# -------------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------
    # YOLO Inference
    # -------------------------------
    results = model(frame, conf=0.4, verbose=False)[0]

    # -------------------------------
    # Process detections
    # -------------------------------
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        label = f"plate {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Draw label background
        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - h - 12),
            (x1 + w + 6, y1),
            (0, 0, 255),
            -1
        )

        # Put label text
        cv2.putText(
            frame,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # -------------------------------
    # Show Output
    # -------------------------------
    cv2.imshow("Car Number Plate Detection (YOLO)", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
