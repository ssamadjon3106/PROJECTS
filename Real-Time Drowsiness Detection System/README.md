# Real-Time Driver Drowsiness Detection System

## Overview
This project is a real-time computer vision system that detects driver drowsiness using facial landmark analysis and Eye Aspect Ratio (EAR).

The system uses MediaPipe FaceMesh to extract eye landmarks and computes EAR to determine whether the eyes are closed for a sustained period. If prolonged eye closure is detected, a drowsiness alert is triggered.

---

## Features
- Real-time webcam processing
- Eye landmark detection using MediaPipe
- Eye Aspect Ratio (EAR) calculation
- Drowsiness detection using threshold + temporal logic
- Visual alert on screen
- Alarm sound system
- FPS display

---

## How It Works
1. Capture video from webcam
2. Detect face landmarks using MediaPipe FaceMesh
3. Extract eye coordinates
4. Compute Eye Aspect Ratio (EAR)
5. If EAR < threshold for N consecutive frames → Trigger alert

---

## Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy
- Pygame (for alarm system)

---

## Installation

Clone repository:

```bash
git clone <your-repo-link>
cd Real-Time-Drowsiness-Detection-System