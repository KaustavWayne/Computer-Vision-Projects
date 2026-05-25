import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
 
# ==========================================================
# LOAD YOLO MODEL
# ==========================================================
 
model = YOLO("best-car.pt")
 
print("Classes:", model.names)
 
# ==========================================================
# SINGLE IMAGE PATH
# ==========================================================
 
image_path = "sample.jpg"   # <-- change image name
image_name = os.path.basename(image_path)
 
# ==========================================================
# YOLO PREDICTION
# ==========================================================
 
results = model.predict(
    image_path,
    conf=0.4,
    imgsz=640,
    verbose=False
)[0]
 
# ==========================================================
# PLOT YOLO RESULT
# ==========================================================
 
annotated_image = results.plot()
 
# Convert BGR to RGB for matplotlib
annotated_image = cv2.cvtColor(
    annotated_image,
    cv2.COLOR_BGR2RGB
)
 
# ==========================================================
# SHOW IMAGE
# ==========================================================
 
plt.figure(figsize=(10, 7))
 
plt.imshow(annotated_image)
 
plt.title(image_name, fontsize=12)
 
plt.axis("off")
 
plt.show()