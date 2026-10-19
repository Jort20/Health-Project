# Health-Project
how different kind of study breaks effects memory retention and focus during learning sessions

**eye_Tracker** is a multimodal data collection tool designed for cognitive focus and attention research. It captures and analyzes facial expressions, gaze patterns, head pose, and ambient audio to classify visual behavior (e.g., fixations vs. saccades) in real-time.

*Note: This project is currently in active development (Work in Progress) and serves as a scientific review build.*

## Features

- **Real-Time Gaze Tracking & Calibration:** Utilizes a 25-point SVR (Support Vector Regression) calibration grid with adaptive EMA smoothing.
- **Cognitive State Analysis:** Hybrid I-VT/I-DT algorithm to classify eye movements into Fixations and Saccades.
- **Facial Metrics:** Tracks Eye Aspect Ratio (EAR) for blink detection, head pose estimation (pitch, yaw, roll), and brow contraction index using MediaPipe.
- **Audio Monitoring:** Calculates ambient audio levels (RMS) to contextualize the environment.
- **Automated Data Logging:** Exports all synchronized features (timestamps, gaze coordinates, ML features, audio RMS) directly to a CSV file.

## Requirements

The tracker is built with Python. The exact package versions required are listed in `requirements.txt`:

```text
opencv-python==5.0.0.93
mediapipe==0.10.14
numpy==2.5.2
sounddevice==0.5.6
scikit-learn==1.9.0

```

## Installation

1. Clone or download this repository.
2. (Optional but recommended) Create a virtual environment for the project.
3. Install the required dependencies:
```bash
pip install -r requirements.txt

```

## Usage

Run the main script to start the tracker:

```bash
python eye_tracker.py

```

**Calibration Phase:**
Upon starting, the application will enter calibration mode. Look at the red dots on the screen and press the **SPACEBAR** to capture your gaze points. Ensure your head pose and distance are within the required thresholds (feedback is displayed on-screen).

**Tracking Phase:**
After calibration, the tracker will automatically start logging data to `focus_data_log.csv` and display a real-time HUD with your cognitive state metrics. Press **'q'** to quit the application and safely save the data.

## License / Authors

Developed by SOLJERS.
