# This is for low specs laptops. 

import cv2
from ultralytics import YOLO

# ==========================================================
# LOAD MODEL
# ==========================================================

model = YOLO("best-car.pt")

print("Classes:", model.names)

# ==========================================================
# OPEN WEBCAM
# ==========================================================

cap = cv2.VideoCapture(0)

# Optional: lower resolution for better FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Webcam started... Press Q to quit")

# ==========================================================
# LIVE DETECTION LOOP
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    # ======================================================
    # YOLO INFERENCE
    # ======================================================

    results = model(frame, verbose=False)[0]

    # ======================================================
    # DRAW ALL DETECTIONS AUTOMATICALLY
    # ======================================================

    annotated_frame = results.plot()

    # ======================================================
    # SHOW FRAME
    # ======================================================

    cv2.imshow(
        "YOLO Live Detection",
        annotated_frame
    )

    # ======================================================
    # EXIT ON Q
    # ======================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================================
# CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()

print("✅ Webcam stopped")


### ========================================================== Second file: live-video-detect.py =================================================

# import cv2
# from ultralytics import YOLO

# # ==========================================================
# # LOAD MODEL
# # ==========================================================

# model = YOLO("best-car.pt")

# print("Classes:", model.names)

# # ==========================================================
# # OPEN WEBCAM
# # ==========================================================

# cap = cv2.VideoCapture(0)

# # Lower resolution for better FPS on weaker laptops
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# # ==========================================================
# # MAIN LOOP
# # ==========================================================

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     # ======================================================
#     # YOLO INFERENCE
#     # ======================================================

#     results = model.predict(
#         frame,
#         verbose=False
#     )[0]

#     # ======================================================
#     # DRAW DETECTIONS
#     # ======================================================

#     for box in results.boxes:

#         x1, y1, x2, y2 = box.xyxy[0]

#         x1, y1, x2, y2 = (
#             int(x1),
#             int(y1),
#             int(x2),
#             int(y2)
#         )

#         conf = float(box.conf[0])

#         cls = int(box.cls[0])

#         label = f"{model.names[cls]} {conf:.2f}"

#         cv2.rectangle(
#             frame,
#             (x1, y1),
#             (x2, y2),
#             (0, 255, 0),
#             2
#         )

#         cv2.putText(
#             frame,
#             label,
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 0),
#             2
#         )

#     # ======================================================
#     # SHOW FRAME
#     # ======================================================

#     cv2.imshow("YOLO Live Detection", frame)

#     # Press Q to quit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # ==========================================================
# # RELEASE RESOURCES
# # ==========================================================

# cap.release()
# cv2.destroyAllWindows()