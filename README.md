# 🎯 Face Recognition System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-View%20Only-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)

**A complete real-time face recognition web application built with OpenCV LBPH algorithm and deployed on Streamlit Cloud.**

[🚀 Live Demo](https://rahul-face-recognition.streamlit.app) • [📁 GitHub](https://github.com/rahul07-github/Face_Recognition_System)

</div>

---

## 📌 Project Overview

This project is a **3-step end-to-end Face Recognition System** built entirely using classical Computer Vision techniques. It allows users to:

1. **Create Dataset** — Capture face images using webcam directly in browser
2. **Train Model** — Train an LBPH face recognizer on the captured dataset
3. **Recognize Faces** — Take a photo and instantly identify the person

The system is deployed as a **live web application** on Streamlit Cloud, accessible from any device with a browser and camera.

> Built as part of B.Tech Final Year Project — SAM Global University, Bhopal  
> Domain: Deep Learning & Computer Vision

---

## 🌐 Live Demo

| Platform | Link |
|---|---|
| 🚀 Streamlit Cloud | [rahul-face-recognition.streamlit.app](https://rahul-face-recognition.streamlit.app) |
| 💻 GitHub Repository | [github.com/rahul07-github/Face_Recognition_System](https://github.com/rahul07-github/Face_Recognition_System) |

---

## 🛠️ Tools & Technologies

| Category | Technology | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core development |
| Computer Vision | OpenCV (cv2) | Face detection + recognition |
| ML Algorithm | LBPH (Local Binary Pattern Histogram) | Face recognition model |
| Face Detection | Haar Cascade Classifier | Frontal + alt2 cascade |
| Web Framework | Streamlit | Interactive web UI |
| Image Processing | NumPy, Pillow | Array ops + image handling |
| Deployment | Streamlit Cloud | Free cloud hosting |
| Version Control | Git + GitHub | Code management |
| OS Packages | libgl1, libglib2.0-0t64 | Linux system dependencies |

---

## 📁 Project Structure

```
Face_Recognition_System/
│
├── app.py                  # Main Streamlit application (3 tabs)
├── face_train.py           # Standalone training script (local use)
├── face_recognition.py     # Standalone recognition script (local use)
├── datasetcreates.py       # Standalone dataset capture (local use)
│
├── face_model.yml          # Trained LBPH model (binary)
├── labels.json             # Label-to-name mapping
│
├── dataset/                # Face image dataset
│   ├── Rahul/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── Nisha/
│   └── Hrisabh/
│
├── requirements.txt        # Python dependencies
├── packages.txt            # Linux system packages (Streamlit Cloud)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## ⚙️ How It Works — Complete Pipeline

```
📸 Webcam Input
      │
      ▼
🔍 Face Detection
   (Haar Cascade — haarcascade_frontalface_default.xml)
   (Fallback: haarcascade_frontalface_alt2.xml)
      │
      ▼
✂️  Face Crop + Preprocessing
   → Convert to Grayscale
   → Histogram Equalization (equalizeHist)
   → Gaussian Blur (3×3)
   → Resize to 100×100 px
      │
      ▼
🧠 LBPH Model
   (Training) → recognizer.train(faces, labels) → face_model.yml
   (Testing)  → recognizer.predict(face) → (label, confidence)
      │
      ▼
📊 Confidence Decision
   confidence < threshold → ✅ Known Person (show name)
   confidence ≥ threshold → ❓ Unknown
      │
      ▼
🖥️ Display Result on Streamlit UI
```

### Confidence Logic (LBPH)
> In LBPH: **lower confidence = better match**
> - `0–60` → Excellent match
> - `60–100` → Good match  
> - `100–120` → Acceptable match
> - `>120` → Unknown person

---

## 🧩 App Features

### Tab 1 — Dataset Create
- Enter person name → Camera opens in browser
- Face auto-detected using Haar Cascade
- Green box = face found, Red box = no face
- Save face with one click
- Unlimited image capture (no fixed limit)
- View + delete individual images
- Progress tracker per person

### Tab 2 — Train Model
- Trains LBPH on all captured persons at once
- Preprocessing: grayscale → equalizeHist → resize
- Saves `face_model.yml` + `labels.json`
- Cache auto-refreshes after training

### Tab 3 — Face Recognition
- Take photo → instant face detection + recognition
- Primary cascade + alt2 cascade fallback
- Confidence bar shown below face box
- Adjustable threshold slider (60–150)
- Shows cropped face + person name + score

---

## ⚡ Local Setup & Installation

### Prerequisites
```
Python 3.9+
Webcam
```

### Step 1 — Clone Repository
```bash
git clone https://github.com/rahul07-github/Face_Recognition_System.git
cd Face_Recognition_System
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ For local use, install `opencv-contrib-python` (not headless):
```bash
pip install opencv-contrib-python
```

### Step 3 — Run App
```bash
streamlit run app.py
```

Open browser → `http://localhost:8501`

---

## 🚀 Deployment — Streamlit Cloud

### Files Required for Cloud Deployment

| File | Purpose |
|---|---|
| `requirements.txt` | Python packages with `opencv-contrib-python-headless` |
| `packages.txt` | Linux system packages (`libgl1`, `libglib2.0-0t64`) |
| `face_model.yml` | Pre-trained model (included in repo) |
| `labels.json` | Label mapping (included in repo) |

### `requirements.txt` (Cloud Version)
```
streamlit>=1.32.0
opencv-contrib-python-headless>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
```

### `packages.txt`
```
libgl1
libglib2.0-0t64
```

> Key difference: Local → `opencv-contrib-python` | Cloud → `opencv-contrib-python-headless`

---

## 🐛 Problems Faced & How They Were Solved

### Problem 1 — `cv2.face` AttributeError
```
AttributeError: module 'cv2' has no attribute 'face'
```
**Cause:** Only `opencv-python` was installed — `cv2.face.LBPHFaceRecognizer_create()` requires the contrib version.  
**Fix:**
```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python
```

---

### Problem 2 — `libgl1-mesa-glx` Package Error on Cloud
```
E: Package 'libgl1-mesa-glx' has no installation candidate
```
**Cause:** Streamlit Cloud uses Debian Trixie — `libgl1-mesa-glx` was renamed.  
**Fix:** Changed `packages.txt` from `libgl1-mesa-glx` → `libgl1`

---

### Problem 3 — `libglib2.0-0` Dependency Conflict
```
libglib2.0-0 : Depends: libffi7 but it is not installable
```
**Cause:** Old package name incompatible with Debian Trixie.  
**Fix:** Changed to `libglib2.0-0t64` (Trixie-compatible version)

---

### Problem 4 — Face Always Showing "Unknown"
**Cause:** Confidence threshold too strict (70) + only 5 training images.  
**Fix:**
- Increased threshold: `70 → 120`
- Recommended minimum 20+ images per person
- Added alt cascade fallback for better detection

---

### Problem 5 — Face Flickering (Unknown ↔ Name)
**Cause:** Single-frame prediction fluctuates with lighting/angle.  
**Fix:** Added majority voting buffer (last 7 frames) in `face_recognition.py`

---

### Problem 6 — Face Not Detected at Angles / Distance
**Cause:** `minNeighbors=7, minSize=(80,80)` was too strict.  
**Fix:**
- `minNeighbors: 7 → 4`
- `minSize: (80,80) → (40,40)`
- Added `haarcascade_frontalface_alt2.xml` as fallback cascade

---

### Problem 7 — Duplicate Slider Widget Crash
```
DuplicateWidgetID Error
```
**Cause:** Two `st.slider()` with same label in `tab_dataset()`.  
**Fix:** Removed duplicate, replaced with `st.number_input()`

---

### Problem 8 — Git Push Error
```
error: src refspec main does not match any
```
**Cause:** `git push` before `git commit` — nothing to push.  
**Fix:** Always: `git add .` → `git commit -m "..."` → `git push`

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | LBPH (Local Binary Pattern Histogram) |
| Face Size | 100 × 100 px |
| Preprocessing | Grayscale + HistEq + GaussianBlur |
| Min Images (recommended) | 20+ per person |
| Detection Cascade | Haar Frontal + Alt2 fallback |
| Confidence Threshold | 120 (adjustable 60–150) |
| Inference Speed | ~50ms per frame |

---


## 📸 Screenshots

**Tab 1 — Dataset Create**

![Dataset Tab](https://github.com/user-attachments/assets/0298b5d8-3ed0-40e4-a939-5093dfe2bd56)

![Dataset Tab 2](https://github.com/user-attachments/assets/65ed7674-5331-402e-aa99-7dc03149da45)

**Tab 2 — Train Model**

![Train Tab](https://github.com/user-attachments/assets/868f1d17-b2b8-4552-b1f4-d47eb0d89c80)

![Train Tab](https://github.com/user-attachments/assets/a50bc756-726e-40c6-9a1e-e90b9c7d0a01)


**Tab 3 — Face Recognition**

![Recognition Tab](https://github.com/user-attachments/assets/250b07dd-c3cc-4821-9e5b-a7fe09f5050c)

---

## 👨‍💻 Author

**Rahul Kumar Jha**  
B.Tech Computer Science (Data Science & GenAI)  
SAM Global University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/rahul07)
[![GitHub](https://img.shields.io/badge/GitHub-rahul07--github-black?style=flat&logo=github)](https://github.com/rahul07-github)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-purple?style=flat)](https://rahul07-github.github.io)

---

## 📜 License — Restricted View Only

```
Copyright (c) 2026 Rahul Kumar Jha. All Rights Reserved.

RESTRICTED LICENSE — VIEW ONLY

This repository and all its contents (code, models, images, documentation)
are the intellectual property of Rahul Kumar Jha.

YOU ARE PERMITTED TO:
  ✅ View and read this code for educational/reference purposes

YOU ARE NOT PERMITTED TO:
  ❌ Copy, clone, or download this code
  ❌ Modify, adapt, or build upon this code
  ❌ Distribute, publish, or share this code
  ❌ Use this code commercially or privately in your own projects
  ❌ Remove this license or attribution

No part of this project may be reproduced, distributed, or transmitted
in any form or by any means without prior written permission from the author.

For permissions or collaborations, contact via LinkedIn or GitHub.
```

---

<div align="center">

Made with ❤️ by **Rahul Kumar Jha** | Gurgram, Haryana, India 🇮🇳

⭐ If you found this project helpful, please consider starring the repository!

</div>
