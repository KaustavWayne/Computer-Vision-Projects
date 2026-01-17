import cv2
import numpy as np
import tensorflow as tf

# ==========================
# Load Haar cascade
# ==========================
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ==========================
# Load VGG16 model
# ==========================
IMG_SIZE = 224
MODEL_PATH = "face_mask_vgg16_model-2.h5"

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ VGG16 model loaded")

# ==========================
# Start webcam
# ==========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        # --------------------
        # Crop face
        # --------------------
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        # --------------------
        # Preprocess
        # --------------------
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        # --------------------
        # Predict
        # --------------------
        prob = model.predict(face, verbose=0)[0][0]

        # --------------------
        # Decision
        # --------------------
        if prob > 0.5:
            label = "With Mask"
            color = (0, 255, 0)
            confidence = prob * 100
        else:
            label = "Without Mask"
            color = (0, 0, 255)
            confidence = (1 - prob) * 100

        text = f"{label} ({confidence:.2f}%)"

        # --------------------
        # Draw
        # --------------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Face Mask Detection (VGG16 + Haar)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
