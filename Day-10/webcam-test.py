import cv2
import math
import cvzone
from ultralytics import YOLO

# ==============================
# START WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not opening")
    exit()

# ==============================
# LOAD YOLO MODEL
# ==============================
model = YOLO("best-robo.pt")

# ==============================
# CLASS NAMES (same order as data.yaml)
# ==============================
classNames = ['With Helmet', 'Without Helmet']

print("✅ Webcam started... Press Q to quit")

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read frame")
        break

    results = model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            # Draw rectangle with corners
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class ID
            cls = int(box.cls[0])

            # Label text
            label = f"{classNames[cls]} {conf}"

            # Put label
            cvzone.putTextRect(
                img,
                label,
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=2
            )

    cv2.imshow("Helmet Detection Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed")
