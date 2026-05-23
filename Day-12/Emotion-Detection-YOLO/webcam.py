import cv2
from ultralytics import YOLO

# ==========================================================
# LOAD FACE DETECTION MODEL
# ==========================================================
# YOLOv8n Detection Model

face_model = YOLO("yolov8n.pt")

# ==========================================================
# LOAD TRAINED EMOTION CLASSIFICATION MODEL
# ==========================================================

emotion_model = YOLO("best.pt")

print("✅ Models Loaded Successfully")

# ==========================================================
# START WEBCAM
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not opening")
    exit()

print("✅ Webcam started... Press ESC to quit")

# ==========================================================
# WEBCAM LOOP
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    # ======================================================
    # PERSON DETECTION
    # ======================================================

    face_results = face_model.predict(
        frame,
        conf=0.5,
        verbose=False
    )[0]

    # ======================================================
    # DRAW YOLO BOXES
    # ======================================================

    annotated_frame = frame.copy()

    # ======================================================
    # LOOP THROUGH DETECTIONS
    # ======================================================

    for box in face_results.boxes:

        # --------------------------------------------------
        # FILTER ONLY PERSON CLASS
        # COCO CLASS 0 = PERSON
        # --------------------------------------------------

        cls = int(box.cls[0])

        if cls != 0:
            continue

        # --------------------------------------------------
        # GET BOX COORDINATES
        # --------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # --------------------------------------------------
        # SAFETY CHECK
        # --------------------------------------------------

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(0, x2)
        y2 = max(0, y2)

        # --------------------------------------------------
        # CROP PERSON REGION
        # --------------------------------------------------

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # ==================================================
        # EMOTION CLASSIFICATION
        # ==================================================

        emotion_results = emotion_model.predict(
            face_crop,
            imgsz=128, # 224
            verbose=False
        )[0]

        # ==================================================
        # GET PREDICTION
        # ==================================================

        pred_index = emotion_results.probs.top1

        confidence = (
            emotion_results.probs.top1conf.item() * 100
        )

        label = emotion_results.names[pred_index]

        text = f"{label} ({confidence:.2f}%)"

        # ==================================================
        # DRAW PERSON BOX
        # ==================================================

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ==================================================
        # DRAW LABEL
        # ==================================================

        cv2.putText(
            annotated_frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # ======================================================
    # SHOW OUTPUT
    # ======================================================

    cv2.imshow(
        "YOLOv8 Emotion Detection",
        annotated_frame
    )

    # ======================================================
    # ESC KEY TO EXIT
    # ======================================================

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================================================
# RELEASE
# ==========================================================

cap.release()

cv2.destroyAllWindows()

print("✅ Webcam Closed")