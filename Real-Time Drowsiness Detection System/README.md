# Real-Time Drowsiness Detection System

A real-time computer vision system that detects driver drowsiness using facial landmarks and Eye Aspect Ratio (EAR).  
Built with MediaPipe and OpenCV, this project analyzes eye closure patterns from webcam input and issues alerts when drowsiness is detected.

---

## 🚀 Features

- 🧠 Real-time webcam feed processing  
- 👁️ Face and eye landmark detection via MediaPipe FaceMesh  
- 📏 Eye Aspect Ratio (EAR) calculation  
- 🚨 Drowsiness detection using configurable threshold logic  
- 🔊 Alarm alert system (optional audio)  
- 📊 On-screen EAR & FPS display

---

## 🔍 How It Works

1. Capture frame from webcam.
2. Detect facial landmarks using MediaPipe FaceMesh.
3. Extract eye landmarks and compute EAR.
4. If EAR remains below threshold for a temporal window → trigger alert.

---

## 🛠️ Installation & Setup

Clone the repository:

```bash
git clone https://github.com/ssamadjon3106/PROJECTS.git
cd "Real-Time Drowsiness Detection System"
