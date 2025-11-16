# Model Training Information

## Overview
This YOLO model was trained to detect basketball-related objects in real-time video footage.

## Training Details

### Dataset
- **Training images**: 9,599 images
- **Validation images**: 873 images  
- **Test images**: 436 images
- **Total dataset**: 10,908 annotated images

### Classes Detected
The model was trained to recognize 5 different classes:
1. **Basketball** (class 0) - The basketball itself
2. **Referee** (class 1) - Game referees
3. **Player** (class 2) - Basketball players
4. **Hoop** (class 3) - Basketball hoops/rings
5. **Ball** (class 4) - Alternative ball detection

### Model Architecture
- **Base model**: YOLOv11 nano (yolo11n.pt)
- **Framework**: Ultralytics YOLO
- **Training output**: best.pt (best performing weights)

### Training Configuration
The model was trained using a `data.yaml` configuration file that specified:
- Paths to training, validation, and test datasets
- Class names and their corresponding IDs
- Image annotations in YOLO format (*.txt files)

### Training Process
1. Started with pre-trained YOLOv11 nano weights
2. Fine-tuned on basketball-specific dataset
3. Validated on separate validation set
4. Best weights saved as `best.pt`

### Results
The trained model (`best.pt`) can now detect basketball players, referees, basketballs, and hoops in real-time video feeds with high accuracy.

## Usage
Run the live detection with:
```bash
python live_camera_detection.py
```

The model performs real-time inference on camera input and displays bounding boxes with class labels and confidence scores.

---
*Model trained for the CourtVision project - Basketball detection and analysis*

