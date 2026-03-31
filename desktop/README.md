# PostureGuard Desktop

This is the standalone Desktop version of the PostureGuard application. 
It bypasses all Web constraints (WebSocket latency, Canvas encoding limits, Tab sleeping) by running Real-Time inference locally using OpenCV and CustomTkinter.

## Requirements
Ensure you are using a Python 3.10+ environment.

```bash
cd desktop
pip install -r requirements.txt
```

## Running the App
```bash
python app.py
```

## Features
- Hardware accelerated OpenCV Camera Capture (Zero-latency)
- Dark-mode User Interface (`customtkinter`)
- Same robust `posture_math.py` calibration and validation logic as the web server.
- No backend required!
