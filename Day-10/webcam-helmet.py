import cv2
from ultralytics import YOLO

# ==============================
# LOAD YOLO MODEL
# ==============================
model = YOLO("best-robo.pt")

# ==============================
# START WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not opening")
    exit()

print("✅ Webcam started... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    # ==============================
    # YOLO PREDICTION
    # ==============================
    results = model.predict(frame, conf=0.4, imgsz=640, verbose=False)[0]

    # ==============================
    # AUTO DRAW BOXES (NO cv2.rectangle manually)
    # ==============================
    annotated_frame = results.plot()

    # ==============================
    # SHOW FRAME
    # ==============================
    cv2.imshow("Helmet Detection - Webcam", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# RELEASE WEBCAM
# ==============================
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed")