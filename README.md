# Eyetrax: Advanced Webcam-Based Eye-Tracking

This eyetracker offers a eye-tracking solution using a standard webcam. It combines computer vision, machine learning, and mathematical filters to accurately record gaze direction, blink frequency, and ambient noise levels.

**Attribution:** 
This project builds upon and utilizes the core architecture of the original EyeTrax software:
> *Zhang, C. (2025). EyeTrax (0.2.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17188537*

---

## Table of Contents
1. [Installation & Virtual Environment](#1-installation--virtual-environment)
2. [Architecture & Modules Explained](#2-architecture--modules-explained)
3. [Data Output (CSV)](#3-data-output-csv)
4. [Running the Application & Command-Line Options](#4-running-the-application--command-line-options)
5. [Project Structure](#5-project-structure)
6. [How to Perform a Calibration](#6-how-to-perform-a-calibration)
7. [Accuracy and Testing](#7-accuracy-and-testing)
8. [Data Collection and Storage](#8-data-collection-and-storage)
9. [Limitations of Webcam-Based Eye-Tracking](#9-limitations-of-webcam-based-eye-tracking)

---

## 1. Installation & Virtual Environment

For Python projects, using a **virtual environment (venv)** is important. It creates an isolated environment on your computer, preventing version conflicts with other projects.

### Step-by-step venv creation

**Step 1: Open your terminal** (Command Prompt/PowerShell on Windows, Terminal on macOS/Linux) and navigate to your project folder.

**Step 2: Create the virtual environment**
Run the following command to create a folder named `venv`:
*   **Windows:** `python -m venv venv`
*   **macOS/Linux:** `python3 -m venv venv`

**Step 3: Activate the virtual environment**
You will know it succeeded when you see `(venv)` at the beginning of your command line prompt.
*   **Windows (Command Prompt):** `venv\Scripts\activate.bat`
*   **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
*   **macOS/Linux:** `source venv/bin/activate`

**Step 4: Install the requirements**
Now that the environment is active, install the exact required versions:
```bash
pip install --upgrade pip
pip install -r requirements.txt

```

*(To exit the environment later, simply type `deactivate` in your terminal).*

---

## 2. Architecture & Modules Explained


### 2.1 The Model (`eyetrax.gaze`)

The core utilizes the `face_landmark.task` model (powered by Google MediaPipe) to detect 478 3D coordinates on the face in real-time.

* **Eye Aspect Ratio (EAR):** By dividing the distance between the upper and lower eyelid by the width of the eye, the algorithm calculates the EAR. If this value drops below a threshold, the system registers a blink, preventing wild cursor jumps.

### 2.2 Filters & Smoothing (`eyetrax.filters`)

Raw eye-tracking data contains noise (jitter). We apply mathematical smoothing to make the data usable:

* **`NoSmoother`:** Passes raw coordinates directly. 
* **`KalmanSmoother`:** The best standard choice. Uses velocity predictions to offer a balance between smooth movement and low lag.
* **`KalmanEMASmoother`:** High UI stability (e.g., typing with eyes). stability at the cost of slight delay during rapid movements.
* **`KDESmoother`:** Kernel Density Estimation. it ignores sudden outliers and shows exactly where the user is focusing.

### 2.3 Calibration Methods (`eyetrax.calibration`)

Calibration maps 3D facial vectors to 2D screen coordinates.

* **`5p` (5-point):** Fast startup, sufficient for simple interactions.
* **`9p` (9-point):** The standard (default). Best balance of speed and high accuracy.
* **`dense`:** Calibrates dozens of points. Best for high-precision scientific research.
* **`lissajous`:** A moving 'figure-eight' target. Best for measuring smooth pursuit.

### 2.4 Utils & External Modules

* **AmbientAudioMonitor:** Runs in an asynchronous background thread to calculate RMS volume and Peak decibels without bottlenecking the video frame rate.

---

## 3. Data Output (CSV)

Once running, a `Data/` folder is automatically created. A CSV file is written in real-time with the following structure:

| Column | Data Type | Explanation |
| --- | --- | --- |
| `timestamp` | Float | UNIX time (millisecond precision) for synchronization. |
| `x_raw`, `y_raw` | Integer | The unfiltered, raw pixel coordinates of the gaze. |
| `x_smooth`, `y_smooth` | Integer | The filtered pixel coordinates (based on your smoother). |
| `blink` | Integer (0/1) | `1` if the model detects closed eyes, `0` if open. |
| `ear` | Float | Eye Aspect Ratio. Lower numbers mean closed eyelids. |
| `sound_db`, `sound_peak_db` | Float | Mean and maximum decibel level in the room. |

---

## 4. Running the Application & Command-Line Options

Once installed and activated, start the application via the CLI:

```bash
eyetrax-virtualcam

```

### Run Options (Flags)

* `--camera <integer>`: Selects webcam (default is `0`).
* `--calibration <method>`: Options are `5p`, `9p` (default), `dense`, `lissajous`.
* `--filter <method>`: Options are `kalman` (default), `kalman_ema`, `kde`, `none`.
* `--model_file <path>`: Bypasses startup calibration by loading a saved profile.

**Example 1 (Standard):** `eyetrax-virtualcam --calibration 9p --filter kalman`
**Example 2 (UI Mode):** `eyetrax-virtualcam --calibration 5p --filter kalman_ema --ema_alpha 0.4`

---

## 5. Project Structure

```text
eyetrax/
│
├── src/eyetrax/
│   ├── app/                # Main application scripts
│   ├── utils/              # Helper functions (video stream, screen logic)
│   ├── calibration.py      # Calibration algorithms
│   ├── filters.py          # Smoothing algorithms (Kalman, KDE, EMA)
│   ├── gaze.py             # ML wrapper and landmark extraction
│   └── cli.py              # CLI argument parsing
│
├── Data/                   # Auto-generated folder for CSV logs
├── pyproject.toml          # Build system configuration
├── requirements.txt        # Explicit Python dependencies
└── README.md               # This documentation

```

---

## 6. How to Perform a Calibration

1. **Setup:** Ensure you are in a well-lit room without strong backlighting. Face the webcam directly.
2. **Start:** Run the command (e.g., `eyetrax-virtualcam --calibration 9p`).
3. **Follow the Target:** A full-screen window opens with a target point.
4. **Keep Head Still:** Move *only* your eyes to look at the target.
5. **Completion:** The window closes automatically once all points are collected, and CSV logging begins.

---

## 7. Accuracy and Testing

* **Accuracy**: The accuracy of the eye-tracking system depends on factors such as lighting conditions, camera resolution, user positioning and calibration quality. Actual spatial accuracy should be established through experimental validation.
* **Testing:** Evaluated dynamically during calibration via the re-projection error (pixel distance between predicted gaze and actual target).
* **Blinks:** The EAR threshold is constantly tested to prevent false-positive movements during blinks.

---

## 8. Data Collection and Storage

All data is stored locally.

* **Storage:** Data is appended real-time to a timestamped CSV file (e.g., `gaze_data_YYYY-MM-DD_HH-MM.csv`) in the `Data/` directory.

---

## 9. Limitations of Webcam-Based Eye-Tracking

* **Head Movement:** Cannot easily distinguish between head turns and eye movements. Keep your head relatively stationary.
* **Lighting:** Low light or strong backlighting heavily degrades iris detection.
* **Obstructions:** Thick glasses, lens glare, or hair over the eyes can cause frame drops or `NaN` values.
