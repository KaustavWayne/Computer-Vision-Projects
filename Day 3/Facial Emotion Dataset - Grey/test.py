# WEBCAM CODE — CUSTOM CNN (GRAYSCALE, SPARSE)

import cv2
import numpy as np
import tensorflow as tf

# ==========================
# Load Haar Cascade
# ==========================
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ==========================
# Load Trained CNN Model
# ==========================
IMG_SIZE = 48   # must match training
MODEL_PATH = "emotion_cnn_sparse_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Grayscale CNN emotion model loaded")

# ==========================
# Class Names (MUST match training order)
# ==========================
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ==========================
# Start Webcam
# ==========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert full frame to grayscale (for face detection + model)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces on grayscale frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        # --------------------
        # Crop face (grayscale)
        # --------------------
        face_gray = gray[y:y+h, x:x+w]

        if face_gray.size == 0:
            continue

        # --------------------
        # Preprocess for CNN
        # --------------------
        face_gray = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))

        # Normalize
        face_array = face_gray.astype("float32") / 255.0

        # Add channel dimension -> (H, W, 1)
        face_array = np.expand_dims(face_array, axis=-1)

        # Add batch dimension -> (1, H, W, 1)
        face_array = np.expand_dims(face_array, axis=0)

        # --------------------
        # Predict Emotion
        # --------------------
        preds = model.predict(face_array, verbose=0)[0]
        pred_index = np.argmax(preds)
        confidence = preds[pred_index] * 100
        label = class_names[pred_index]

        text = f"{label} ({confidence:.2f}%)"

        # --------------------
        # Draw Results
        # --------------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Detection (CNN Grayscale)", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
