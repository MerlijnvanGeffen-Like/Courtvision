# YOLO Live Camera Detection

Live camera detection using your trained YOLO model (`best.pt`).

## Quick Start

### 1. Install Dependencies

Run the batch file to install all required packages:
```bash
install_requirements.bat
```

Or manually install:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python
```

### 2. Run Live Detection

```bash
python live_camera_detection.py
```

## Controls

- **'q'** - Quit the application
- **'s'** - Save current frame as snapshot

## Features

- ✅ Real-time object detection using your trained YOLO model
- ✅ Live FPS display
- ✅ Automatic bounding boxes and labels
- ✅ Mirror mode (flipped camera for natural viewing)
- ✅ Save snapshots on demand

## Files

- `best.pt` - Your trained YOLO model (best weights)
- `last.pt` - Last checkpoint from training
- `yolo11n.pt` - Base YOLO11 model
- `live_camera_detection.py` - Main application
- `requirements.txt` - Python dependencies

## Troubleshooting

### PyTorch DLL Error

If you get a DLL error, install the CPU version of PyTorch:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Camera Not Found

Make sure your camera is:
1. Connected properly
2. Not being used by another application
3. Drivers are installed

Try changing the camera index in the code if you have multiple cameras.

### Model Not Loading

Ensure `best.pt` exists in the same directory as the script.

## Model Information

Your model was trained to detect specific objects. The detection results will show:
- Bounding boxes around detected objects
- Class names
- Confidence scores

Enjoy your live YOLO detection! 🎥
