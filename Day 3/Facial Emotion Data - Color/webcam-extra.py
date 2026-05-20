import cv2
import numpy as np
import tensorflow as tf

# ==========================================================
# LOAD HAAR CASCADE
# ==========================================================
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

# ==========================================================
# LOAD TRAINED RGB MODEL
# ==========================================================
IMG_SIZE = 224
MODEL_PATH = 'emotion_rgb_model.h5'

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ RGB Emotion Model Loaded")

# ==========================================================
# CLASS NAMES
# MUST match training order
# ==========================================================
class_names = [
    'surprise',
    'fear',
    'angry',
    'neutral',
    'sad',
    'disgust',
    'happy'
]

# ==========================================================
# START WEBCAM
# ==========================================================
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ======================================================
    # Convert full frame to grayscale
    # ONLY for face detection
    # ======================================================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ======================================================
    # Detect faces
    # ======================================================
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        # ==================================================
        # Crop COLOR face from original frame
        # ==================================================
        face_color = frame[y:y+h, x:x+w]

        if face_color.size == 0:
            continue

        # ==================================================
        # Resize
        # ==================================================
        face = cv2.resize(
            face_color,
            (IMG_SIZE, IMG_SIZE)
        )

        # ==================================================
        # Convert BGR → RGB
        # ==================================================
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # ==================================================
        # Normalize
        # SAME as training
        # ==================================================
        face_array = face.astype("float32") / 255.0

        # ==================================================
        # Add batch dimension
        # (1, H, W, 3)
        # ==================================================
        face_array = np.expand_dims(
            face_array,
            axis=0
        )

        # ==================================================
        # PREDICT EMOTION
        # ==================================================
        preds = model.predict(
            face_array,
            verbose=0
        )[0]

        pred_index = np.argmax(preds)

        confidence = preds[pred_index] * 100

        label = class_names[pred_index]

        text = f"{label} ({confidence:.2f}%)"

        # ==================================================
        # DRAW RESULTS
        # ==================================================
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

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
        "Emotion Detection (RGB Model)",
        frame
    )

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================================================
# RELEASE
# ==========================================================
cap.release()

cv2.destroyAllWindows()