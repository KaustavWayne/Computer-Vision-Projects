import os
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

# ==========================================================
# CONFIGURATION
# ==========================================================
INPUT_FOLDER = "/content/input_images"   
OUTPUT_FOLDER = "/content/output_images" 

# ==========================================================
# LOAD MODELS
# ==========================================================
print("🔄 Loading YOLO model...")
model = YOLO("best-car.pt")
names = model.names
print(f"✅ Classes: {names}")

print("🔄 Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=True)
print("✅ EasyOCR ready\n")

# ==========================================================
# STREAM DETECTION WITH NATIVE YOLO SAVE PATHS
# ==========================================================
print(f"📂 Processing folder: {INPUT_FOLDER}\n")

results = model.predict(
    source=INPUT_FOLDER,
    conf=0.4,
    imgsz=640,
    verbose=False,
    stream=True,          # Stream one-by-one to keep memory usage low
    save=True,             # REQUIRED: Triggers YOLO to use project/name paths
    project=OUTPUT_FOLDER, # Directs YOLO to your output folder
    name="",               # Drops the extra default 'predict/' subfolder
    exist_ok=True         # Overwrites files if they already exist
)

for idx, result in enumerate(results, start=1):
    image_name = os.path.basename(result.path)
    
    print("-" * 50)
    print(f"🖼️  [{idx}] Processing: {image_name}")
    print("-" * 50)

    # ------------------------------------------------------
    # WHERE THE MAGIC HAPPENS: Fetch YOLO's newly saved image
    # ------------------------------------------------------
    yolo_saved_path = os.path.join(OUTPUT_FOLDER, image_name)
    
    # Read the image YOLO just saved (it already has the default boxes drawn)
    annotated_image = cv2.imread(yolo_saved_path)
    if annotated_image is None:  # <--- FIXED TYPO HERE
        continue

    # Keep a copy of the raw image matrix from RAM for clean OCR cropping
    img_raw = result.orig_img
    
    num_detections = len(result.boxes)
    print(f"📋 Detections: {num_detections}")

    summary_data = []

    # ======================================================
    # OCR PROCESS LOOP
    # ======================================================
    for i, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_score = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = names.get(class_id, "License_Plate")

        # Crop plate snippet from the clean raw image
        roi = img_raw[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Preprocess snippet
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Extract Text
        ocr_result = reader.readtext(thresh)
        best_text = "Not Found"
        ocr_score = 0.0

        if len(ocr_result) > 0:
            best_match = max(ocr_result, key=lambda x: x[2])
            best_text = best_match[1]
            ocr_score = best_match[2]
            
            print(f"CR raw: '{best_text}' | score: {ocr_score:.2f}")
            print(f"    ✅ Plate text: {best_text}")
            print(f"  🖼️  Overlay drawn for '{best_text}' (detection #{i})")
        else:
            print(f"CR raw: 'Not Found' | score: 0.00")

        summary_data.append((i, class_name, conf_score, best_text))

        # Render custom white-box overlays if plate text was found
        if best_text != "Not Found":
            # Draw text right above the bounding box
            cv2.putText(annotated_image, best_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Dynamic White-box generation
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            
            (text_width, text_height), _ = cv2.getTextSize(best_text, font_face, font_scale, thickness)
            box_width = max(300, text_width + 40)
            box_height, plate_height = 80, 120
            
            enlarged_plate = cv2.resize(roi, (box_width, plate_height))
            white_box = 255 * np.ones((box_height, box_width, 3), dtype=np.uint8)
            
            text_x = (box_width - text_width) // 2
            text_y = (box_height + text_height) // 2 - 5
            cv2.putText(white_box, best_text, (text_x, text_y), font_face, font_scale, (0, 0, 0), thickness)

            overlay_x, overlay_y = 50, 50
            if (overlay_y + box_height + plate_height) < annotated_image.shape[0] and \
               (overlay_x + box_width) < annotated_image.shape[1]:
                annotated_image[overlay_y : overlay_y + box_height, overlay_x : overlay_x + box_width] = white_box
                annotated_image[overlay_y + box_height : overlay_y + box_height + plate_height, overlay_x : overlay_x + box_width] = enlarged_plate

    # Overwrite YOLO's saved file with our updated custom OCR overlay image
    cv2.imwrite(yolo_saved_path, annotated_image)
    print(f"\n✅ Updated output saved natively to: {yolo_saved_path}\n")

    # IMAGE CONSOLE REPORT
    print("==================================================")
    print(f"📋 FINAL SUMMARY FOR {image_name}")
    print("==================================================")
    print(f"  #    {'Class':<18} {'Conf':<8} {'Plate Text'}")
    print("--------------------------------------------------")
    for row in summary_data:
        print(f"  {row[0]:<4} {row[1]:<18} {row[2]:<8.2f} {row[3]}")
    print("==================================================\n")

print("🏁 All folder streams processed successfully!")