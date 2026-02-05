import cv2
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, img_path):
        result = self.model(img_path, conf=0.4)[0]
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append((x1, y1, x2, y2, conf))

        return detections

    def draw_boxes(self, img_path, detections):
        img = cv2.imread(img_path)

        for x1, y1, x2, y2, conf in detections:
            label = f"plate {conf:.2f}"

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 4)

            (w,h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3
            )
            cv2.rectangle(
                img, (x1, y1-h-20),
                (x1+w+8, y1),
                (0,0,255), -1
            )

            cv2.putText(
                img, label, (x1+4, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255,255,255), 3
            )

        return img
