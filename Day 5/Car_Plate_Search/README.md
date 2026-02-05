# 🚗 Car Number Plate Detection & Search

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A **YOLO-based Car Number Plate Detection system** with an interactive **Streamlit UI** that allows users to:

- Run inference on image directories or uploaded images  
- Save & reload inference metadata  
- Search and visualize detected plates  
- Display bounding boxes with confidence scores  

---

## 📌 Project Features

- 🔍 YOLO-based number plate detection  
- 🖼️ Directory-based & upload-based inference  
- 📦 Metadata generation (`metadata.json`)  
- 🔎 Search-driven visualization (no auto-render)  
- 📊 Adjustable grid layout  
- 🎨 Styled Streamlit UI  
- 💾 Reusable inference results  

---

## 🧠 High-Level Workflow

(Training → Inference → Metadata Generation → Search & Visualization via Streamlit)

---

## 📁 Project Structure

```text
car_number_plate_detection/
│
├── app.py
├── requirements.txt
│
├── models/
│   └── best.pt
│
├── data/
│   └── uploads/
│
└── src/
    ├── detector.py
    ├── metadata.py
    └── utils.py
```

---

## 🗂️ Dataset

- Source: **Roboflow**
- Format: **YOLOv8**
- Class: `number_plate`

You should download the dataset in **YOLO format**.

---

## 🧪 Model Training (Google Colab)

### Steps

1. Upload dataset ZIP to Google Drive  
2. Mount Drive  
3. Train YOLO model  
4. Save `best.pt` back to Drive  

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)
```

Download the trained weights from:

```text
runs/detect/train/weights/best.pt
```

---

## 💻 Local Setup (VS Code)

### 1️⃣ Create environment

```bash
conda create -n plate python=3.9
conda activate plate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add trained model

Place your trained weights at:

```text
models/best.pt
```

---

## ▶️ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🖥️ Streamlit Usage

### Option 1: Process New Images

- Load from directory path **or** upload new images  
- Click **Run Inference**

You will see a message like:

> Successfully processed X images

---

### Option 2: Search Images

- Click **Search Images**  
- View detected plates with bounding boxes  

---

### Option 3: Load Existing Metadata

- Upload `metadata.json`  
- Instantly visualize previous results  

---

## 📊 Metadata Format (`metadata.json`)

```json
{
  "image_path.jpg": {
    "plate_count": 1,
    "detections": [[x1, y1, x2, y2, 0.82]]
  }
}
```

---

## 🧠 Technologies Used

- Python  
- YOLO (Ultralytics)  
- OpenCV  
- Streamlit  
- Roboflow  
- Google Colab  

---

## 🎯 Interview Talking Point

> “I built a modular computer vision pipeline separating training, inference, metadata persistence, and search-based visualization using Streamlit.”

---

## 🚀 Future Improvements

- OCR for number plate text extraction  
- Video & webcam inference  
- Confidence threshold slider  
- Deployment on Streamlit Cloud  

---

## 📜 License

**MIT License**

---

## ✅ WHAT YOU NOW HAVE

✔ Professional README  
✔ Shields.io badges  
✔ Clear workflow  
✔ Project template  
✔ Interview-ready documentation  

---

