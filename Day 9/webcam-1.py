# Use PaddleOCR

import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ==============================
# LOAD YOLO MODEL
# ==============================
model = YOLO("best.pt")

# ==============================
# LOAD PADDLE OCR
# ==============================
ocr = PaddleOCR(use_angle_cls=True, lang="en")

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

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # OCR (NO cls=True)
        ocr_result = ocr.ocr(roi_rgb)

        plate_text = ""

        if ocr_result and len(ocr_result[0]) > 0:
            plate_text = ocr_result[0][0][1][0]

        if plate_text != "":
            cv2.putText(
                annotated_frame,
                plate_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

    cv2.imshow("YOLO + PaddleOCR Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed")
