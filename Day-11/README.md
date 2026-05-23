# YOLOv8 Android TFLite Object Detection App Guide

## Repository Links

Previous Repository:

https://github.com/surendramaran/YOLOv8-TfLite-Object-Detector

Main Repository:

https://github.com/surendramaran/YOLO/tree/main/YOLOv8-Object-Detector-Android-Tflite

---

# YOLOv8 Live Object Detection Android Application

## Description

This Android application is designed to perform live object detection using the YOLOv8 machine learning model.

YOLOv8 (You Only Look Once version 8) is known for its real-time object detection capabilities, and this app brings that functionality to Android devices.

---

# STEP 1 — DOWNLOAD / CLONE REPOSITORY

Clone or download the repository:

```bash
git clone https://github.com/surendramaran/YOLO.git
```

Open folder:

```text
YOLO/YOLOv8-Object-Detector-Android-Tflite
```

in Android Studio.

---

# STEP 2 — ADD YOUR MODEL

Go to:

```text
app/src/main/assets
```

Replace the existing model with your custom `.tflite` model.

Rename your model to:

```text
yolov8n_int8.tflite
```

Example:

```text
helmet.tflite
→ rename →
yolov8n_int8.tflite
```

---

# STEP 3 — EDIT LABEL FILE

Inside assets folder open:

```text
example_label_file.txt
```

Replace with your class names.

Example for pothole detection:

```text
Pothole
```

Example for helmet detection:

```text
Helmet
No Helmet
```

---

# STEP 4 — EDIT Constants.kt

Open:

```text
app/src/main/java/com/surendramaran/yolov8tflite/Constants.kt
```

Edit like this:

```kotlin
package com.surendramaran.yolov8tflite

object Constants {

    const val MODEL_PATH = "yolov8n_int8.tflite"

    val LABELS_PATH: String? = "example_label_file.txt"
}
```

---

# STEP 5 — CHANGE APP NAME

Open:

```text
app/src/main/res/values/strings.xml
```

Example:

```xml
<resources>
    <string name="app_name">RoadEye AI</string>
    <string name="gpu">GPU</string>
</resources>
```

You can change:

```xml
RoadEye AI
```

to any app name.

Examples:

```text
HelmetGuard AI
FireDetect AI
MaskShield AI
GarbageDetect AI
```

---

# STEP 6 — CHANGE APP LOGO

## STEP 6.1 — SAVE IMAGE

Save your generated logo PNG.

Example:

```text
D:\My-Apks\roadeye_logo.png
```

---

## STEP 6.2 — OPEN IMAGE ASSET TOOL

Right click:

```text
app
```

Then:

```text
New
→ Image Asset
```

---

## STEP 6.3 — CONFIGURE ICON

Choose:

```text
Launcher Icons (Adaptive and Legacy)
```

---

## STEP 6.4 — SELECT YOUR LOGO

Under:

```text
Foreground Layer
```

Click folder icon.

Choose your PNG logo.

---

## STEP 6.5 — ADJUST ICON

Use:

```text
Scaling
```

slider.

Recommended:

```text
80–85%
```

Goal:

- full text visible
- pothole visible
- camera visible
- not touching edges

---

## STEP 6.6 — FINISH

Click:

```text
Next
→ Finish
```

Android Studio automatically creates all icon sizes.

---

# STEP 7 — RUN APP

Connect your Android phone with USB debugging enabled.

Then click:

```text
▶ Run App
```

Android Studio installs app directly to your phone.

---

# STEP 8 — GENERATE APK

Go to:

```text
Build
→ Generate Signed Bundle / APK
→ APK
→ release
```

Then click:

```text
Create new...
```

---

# STEP 9 — CREATE KEYSTORE

Example values:

| Field              | Example                  |
| ------------------ | ------------------------ |
| Key store path     | `D:\My-Apks\pothole.jks` |
| Password           | `123456`                 |
| Confirm Password   | `123456`                 |
| Alias              | `pothole`                |
| Key password       | `123456`                 |

Then click:

```text
Next
→ release
→ Create
```

---

# APK LOCATION

Generated APK usually appears in:

```text
app/release/app-release.apk
```

You can rename it manually:

```text
RoadEye-AI.apk
```

---

# TEMPLATE WORKFLOW FOR FUTURE PROJECTS

After creating first app:

Create clean template by deleting:

```text
.gradle
.idea
app/build
app/release
```

Keep only source code.

Rename folder:

```text
YOLO-Android-Template
```

---

# FUTURE PROJECT WORKFLOW

For future apps:

1. Copy template folder
2. Rename folder
3. Replace model
4. Replace labels
5. Change app name
6. Change logo
7. Build APK

Examples:

```text
HelmetGuard-AI
FireDetect-AI
MaskShield-AI
```

---

# SAFE TO DELETE FOR STORAGE SAVING

Safe to delete:

```text
app/build
.gradle
.idea
```

Android Studio recreates them automatically later.

---

# Contributing

Contributions are welcome!  
Feel free to fork the repository and submit pull requests.

---

# Contact

For questions or feedback, open an issue in the repository.

---

# Support

If you find this project helpful and want to support its development, consider becoming a patron on Patreon.

Thank you!
