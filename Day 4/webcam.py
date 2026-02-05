import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224   # same as training

# -------------------------------
# Load trained model
# -------------------------------
model = tf.keras.models.load_model("bbox_resnet_model_upd.h5")
# or if model already exists, skip loading

print("✅ Bounding Box model loaded")

# -------------------------------
# Start Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # -------------------------------
    # Preprocess frame
    # -------------------------------
    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # -------------------------------
    # Predict bbox
    # -------------------------------
    pred = model.predict(img_input, verbose=0)[0]

    # Fix bbox ordering
    x1, y1, x2, y2 = pred
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Convert normalized -> pixel
    x_min = int(x_min * w)
    y_min = int(y_min * h)
    x_max = int(x_max * w)
    y_max = int(y_max * h)

    # Clip to frame size
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    # -------------------------------
    # Draw bbox
    # -------------------------------
    if x_max > x_min and y_max > y_min:
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, "Detected", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # -------------------------------
    # Show Output
    # -------------------------------
    cv2.imshow("Webcam Bounding Box Detection (ResNet50)", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
