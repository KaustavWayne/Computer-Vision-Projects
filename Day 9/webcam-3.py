# Use Terrasact

import cv2
import pytesseract
from ultralytics import YOLO

# ==============================
# SET TESSERACT PATH (WINDOWS)
# ==============================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ==============================
# LOAD YOLO MODEL
# ==============================
model = YOLO("best.pt")   # your license plate YOLO model

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
    # YOLO DETECTION
    # ==============================
    results = model.predict(frame, conf=0.4, imgsz=640, verbose=False)[0]

    # Auto draw YOLO boxes
    annotated_frame = results.plot(conf=False, labels=False)

    # ==============================
    # OCR FOR EACH DETECTED PLATE
    # ==============================
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # ==============================
        # PREPROCESS FOR OCR
        # ==============================
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # ==============================
        # TESSERACT OCR
        # ==============================
        text = pytesseract.image_to_string(
            thresh,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        text = text.strip()

        # ==============================
        # DRAW OCR TEXT
        # ==============================
        if text != "":
            cv2.putText(
                annotated_frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

    # ==============================
    # SHOW OUTPUT
    # ==============================
    cv2.imshow("YOLO + Tesseract OCR Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed")
