# Easy-OCR Method - . use numpy<2

import cv2
import easyocr
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Load EasyOCR
reader = easyocr.Reader(["en"], gpu=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not opening")
    exit()

print("✅ Webcam started... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO prediction
    results = model.predict(frame, conf=0.4, imgsz=640, verbose=False)[0]

    # Auto draw YOLO boxes
    annotated_frame = results.plot(conf=False, labels=True)

    # OCR on each detected bbox
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ocr_result = reader.readtext(thresh)

        if len(ocr_result) > 0:
            text = ocr_result[0][1]

            # Put OCR text above bbox
            cv2.putText(
                annotated_frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

    cv2.imshow("YOLO + EasyOCR Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed")
