# Football Dataset Directory Structure

This directory contains the training and validation splits for fine-tuning custom YOLO models on football match video frames.

### Directory Structure:
```
data/
├── dataset/
│   ├── images/
│   │   ├── train/          # Training frame images (.jpg / .png)
│   │   └── val/            # Validation frame images (.jpg / .png)
│   ├── labels/
│   │   ├── train/          # YOLO annotation labels (.txt normalized: class x_c y_c w h)
│   │   └── val/            # Validation annotation labels (.txt)
│   └── data.yaml           # Dataset configuration file
└── videos/
    └── sample_broadcast.mp4 # Full 1080p match test stream (750 frames)
```

### Pretrained Weights vs Training Split:
- The active pipeline utilizes **Pretrained YOLOv8 Weights** (`models/yolov8m.pt`), which were pre-trained on 330,000+ images from COCO/SoccerNet.
- This directory is structured to support fine-tuning on proprietary club footage or customized camera angles.
