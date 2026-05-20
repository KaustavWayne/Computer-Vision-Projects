# Webcam

import cv2
import numpy as np
import tensorflow as tf

# ==========================
# Load Haar Cascade
# ==========================
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ==========================
# Load Trained VGG16 Model
# ==========================
IMG_SIZE = 224
MODEL_PATH = "emotion_vgg16_sparse_model.h5"   # your VGG16 model

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ VGG16 emotion model loaded")

# ==========================
# Class Names (MUST match training order!)
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

    # --------------------
    # Convert full frame to grayscale for face detection only
    # --------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------
    # Detect faces on grayscale frame
    # --------------------
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        # --------------------
        # Crop face (COLOR, BGR)
        # --------------------
        face_color = frame[y:y+h, x:x+w]   # BGR crop

        if face_color.size == 0:
            continue

        # --------------------
        # Preprocess for VGG16 (rescale = 1./255 version)
        # --------------------
        # Resize
        face = cv2.resize(face_color, (IMG_SIZE, IMG_SIZE))

        # Convert BGR -> RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)   # (224, 224, 3)

        # Normalize (SAME as training)
        face_array = face.astype("float32") / 255.0

        # Add batch dimension -> (1, 224, 224, 3)
        face_array = np.expand_dims(face_array, axis=0)

        # --------------------
        # Predict Emotion
        # --------------------
        preds = model.predict(face_array, verbose=0)[0]

        """
        preds shape = (NUM_CLASSES,)
        example -> [0.02, 0.01, 0.05, 0.78, 0.04, 0.06, 0.04]

        pred_index = argmax(preds)
        confidence = preds[pred_index] * 100
        label = class_names[pred_index]
        """

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

    # --------------------
    # Show Frame
    # --------------------
    cv2.imshow("Emotion Detection (VGG16 RGB)", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()