import cv2
import math
import cvzone
from ultralytics import YOLO

# ==============================
# VIDEO CAPTURE
# ==============================
video_path = "3.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video not opening")
    exit()

# ==============================
# LOAD YOLO MODEL
# ==============================
model = YOLO("best-robo.pt")

# ==============================
# CLASS NAMES
# ==============================
classNames = ['With Helmet', 'Without Helmet']

print("✅ Video started... Press Q to quit")

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            label = f"{classNames[cls]} {conf}"

            cvzone.cornerRect(img, (x1, y1, w, h))

            cvzone.putTextRect(
                img,
                label,
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=2
            )

    cv2.imshow("Helmet Detection - Video", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Video closed")
