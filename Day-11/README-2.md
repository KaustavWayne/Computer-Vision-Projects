# TEMPLATE WORKFLOW FOR FUTURE PROJECTS

After creating your first app:

Create a clean reusable template by deleting:

```text
.gradle
.idea
app/build
app/release
```

Keep only source code and project structure.

Rename folder:

```text
YOLO-Android-Template
```

---

# Creating a New Android YOLO Project From Template

To create a new project (like Pothole Detection) from your template, follow these steps carefully.

Since you have already unchecked:

```text
Click Project- three dot , Appearance and unchecked the Compact Middle Packages
```

the process becomes much easier.

---

# STEP 1 — Rename the Java Package (Automatic Refactor)

This step changes the internal package name of the app and automatically updates all project files.

## Navigate To:

```text
app
→ src
→ main
→ java
→ com
→ kaustav
```

---

## Then:

1. Right click:

```text
helmetguardai
```

2. Select:

```text
Refactor
→ Rename...
```

3. In popup click:

```text
Rename package
```

4. Type your new project name.

Example:

```text
potholedetector
```

5. Click:

```text
Refactor
```

6. If another popup appears at bottom:

```text
Do Refactor
```

Click it.

---

# WHAT THIS AUTOMATICALLY FIXES

Android Studio automatically updates:

- Kotlin/Java files
- MainActivity path
- Manifest references
- XML references
- Imports
- Package declarations

This prevents errors like:

```text
Activity class does not exist
```

---

# STEP 2 — Update App Identity (Gradle)

This step tells Android that this is a completely separate app.

Without this step, your new app may overwrite older apps on your phone.

---

## Open:

```text
app/build.gradle.kts
```

---

## Update These Two Lines

```kotlin
namespace = "com.kaustav.potholedetector"

applicationId = "com.kaustav.potholedetector"
```

---

## Then Click

```text
Sync Now
```

at the top right.

---

# STEP 3 — Change App Name (Shown On Phone)

This changes the visible app name under the app icon.

---

## Open:

```text
app/src/main/res/values/strings.xml
```

---

## Change:

```xml
<string name="app_name">Pothole Detector</string>
```

---

# STEP 4 — Update Model Files

## Go To:

```text
app/src/main/assets
```

---

## Replace:

- `.tflite` model
- label file

Example:

```text
best_int8.tflite
example_label_file.txt
```

---

# STEP 5 — Update Constants.kt

## Open:

```text
Constants.kt
```

---

## Update Model Name

Example:

```kotlin
const val MODEL_PATH = "best_int8.tflite"
val LABELS_PATH: String? = "example_label_file.txt"
```

---

# STEP 6 — Change App Logo

## Right Click:

```text
app
→ New
→ Image Asset
```

---

## Select:

```text
Launcher Icons (Adaptive and Legacy)
```

Choose your PNG logo.

Recommended scaling:

```text
80–85%
```

Then click:

```text
Next
→ Finish
```

---

# STEP 7 — Clean and Rebuild

Go to:

```text
Build
→ Clean Project
```

Then:

```text
Build
→ Rebuild Project
```

---

# STEP 8 — Run App

Connect your Android phone with USB Debugging enabled.

Then click:

```text
▶ Run App
```

Android Studio installs app directly to your phone.

---

# STEP 9 — Generate APK

Go to:

```text
Build
→ Generate Signed Bundle / APK
→ APK
→ release
```

---

# SUMMARY

## STEP 1

Fixes all package/code references automatically.

Prevents:

```text
Activity class does not exist
```

errors.

---

## STEP 2

Allows multiple apps to exist together on your phone.

Examples:

- HelmetGuard AI
- Pothole Detector
- FireDetect AI

without overwriting each other.

---

## STEP 3

Changes the visible app name under the app icon.

---

# FINAL RESULT

You now have a completely separate Android YOLO application project using the same template.

---

# FUTURE PROJECT WORKFLOW

For future apps:

1. Copy template folder
2. Rename folder
3. Refactor package name
4. Change namespace
5. Change applicationId
6. Replace model
7. Replace labels
8. Change app name
9. Change logo
10. Build APK

Examples:

```text
HelmetGuard-AI
FireDetect-AI
MaskShield-AI
```
