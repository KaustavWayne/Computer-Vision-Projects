import cv2
from ultralytics import YOLO

# ==========================================================
# LOAD HAAR CASCADE FACE DETECTOR
# ==========================================================

face_detector = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

# ==========================================================
# LOAD TRAINED YOLO EMOTION MODEL
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

    # ======================================================
    # READ FRAME
    # ======================================================

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    # ======================================================
    # CONVERT TO GRAYSCALE
    # ======================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # ======================================================
    # FACE DETECTION
    # ======================================================

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # ======================================================
    # LOOP THROUGH DETECTED FACES
    # ======================================================

    for (x, y, w, h) in faces:

        # --------------------------------------------------
        # SAFETY CLIPPING
        # --------------------------------------------------

        x = max(0, x)
        y = max(0, y)

        # --------------------------------------------------
        # CROP FACE FROM GRAYSCALE IMAGE
        # --------------------------------------------------

        face_crop = gray[
            y:y+h,
            x:x+w
        ]

        # --------------------------------------------------
        # EMPTY CHECK
        # --------------------------------------------------

        if face_crop.size == 0:
            continue

        # --------------------------------------------------
        # CONVERT GRAY -> BGR
        # YOLO expects 3 channels
        # --------------------------------------------------

        face_crop = cv2.cvtColor(
            face_crop,
            cv2.COLOR_GRAY2BGR
        )

        # ==================================================
        # EMOTION PREDICTION
        # ==================================================

        results = emotion_model.predict(
            face_crop,
            imgsz=128,
            verbose=False
        )[0]

        # ==================================================
        # GET PREDICTION
        # ==================================================

        pred_index = results.probs.top1

        confidence = (
            results.probs.top1conf.item() * 100
        )

        label = results.names[pred_index]

        text = f"{label} ({confidence:.2f}%)"

        # ==================================================
        # DRAW FACE BOX
        # ==================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # ==================================================
        # DRAW LABEL
        # ==================================================

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # ======================================================
    # SHOW OUTPUT
    # ======================================================

    cv2.imshow(
        "YOLO Emotion Detection",
        frame
    )

    # ======================================================
    # PRESS ESC TO EXIT
    # ======================================================

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================================================
# RELEASE
# ==========================================================

cap.release()

cv2.destroyAllWindows()

print("✅ Webcam Closed")


## Final Recommendation
# Best For YOUR FER Model

# ✅ Haar Cascade
# ✅ Grayscale Face Crop
# ✅ Gray → BGR Conversion
# ✅ YOLO Emotion Classification

# This is the most correct setup based on your dataset.