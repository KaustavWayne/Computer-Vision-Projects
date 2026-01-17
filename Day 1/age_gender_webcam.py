import cv2
import numpy as np
import tensorflow as tf
# --------------------
# Settings
# --------------------
IMG_SIZE = 128
MODEL_PATH = "age_gender_vgg16_model_fix.h5"

# --------------------
# Load model
# --------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded")

# --------------------
# Load face detector
# --------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------
# Start webcam
# --------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🎥 Webcam started. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # crop face
        face = frame[y:y+h, x:x+w]

        # preprocess
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        # predict
        gender_pred, age_pred = model.predict(face, verbose=0)

        gender = "Female" if gender_pred[0][0] > 0.5 else "Male"

        age = int(np.clip(age_pred[0][0], 0, 100))

        label = f"{gender}, {age} yrs"

        # draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # put text
        cv2.putText(
            frame, label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

    cv2.imshow("Age & Gender Detection", frame)

    # press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
