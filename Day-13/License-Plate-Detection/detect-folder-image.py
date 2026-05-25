import os
from ultralytics import YOLO

# ==========================================================
# LOAD YOLO MODEL
# ==========================================================

model = YOLO("best-car.pt")

print("Classes:", model.names)

# ==========================================================
# FOLDER PATHS
# ==========================================================

input_folder  = "images"          # <-- folder with input images
output_folder = "images_output"   # <-- folder to save results

# ==========================================================
# YOLO PREDICTION ON FOLDER
# ==========================================================

results = model.predict(
    source=input_folder,
    conf=0.4,
    imgsz=640,
    verbose=True,
    save=True,                    # save annotated images
    project=output_folder,        # output folder
    name="",                      # no extra subfolder
    exist_ok=True                 # overwrite if folder exists
)

# ==========================================================
# DONE
# ==========================================================

print(f"\n✅ Detection Completed!")
print(f"✅ Results saved to: {output_folder}")
print(f"✅ Total images processed: {len(results)}")