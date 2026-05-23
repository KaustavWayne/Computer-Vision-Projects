import cv2
from ultralytics import YOLO

# ==========================================================
# LOAD HAAR CASCADE FACE DETECTOR
# ==========================================================

# face_detector = cv2.CascadeClassifier(
#     cv2.data.haarcascades +
#     'haarcascade_frontalface_default.xml'
# )

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
        # CROP FACE
        # --------------------------------------------------

        face_crop = frame[
            y:y+h,
            x:x+w
        ]

        if face_crop.size == 0:
            continue

        # ==================================================
        # EMOTION PREDICTION
        # ==================================================

        results = emotion_model.predict(
            face_crop,
            imgsz=128, # 224
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
        "Emotion Detection",
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



#### Your Haar Cascade version is the proper implementation for your emotion project. ###