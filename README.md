# Courtvision

Basketball tracking system that uses YOLO computer vision to detect shots, track scores, and calculate accuracy in real-time. Includes a React web app for viewing statistics, leaderboards, and managing settings.

## Installation

```bash
pip install -r requirements.txt
```

## Run
```bash
run.bat
```
OR
```bash
python live_camera_detection.py
```

## Controls

- **'q'** - Quit
- **'s'** - Save screenshot

## Troubleshooting

**DLL Error**: Install CPU version of PyTorch:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Camera Not Found**: Ensure camera is connected and not used by another app.

**Model Not Loading**: Ensure `best.pt` exists in the same directory.
