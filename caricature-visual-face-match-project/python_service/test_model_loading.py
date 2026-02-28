#!/usr/bin/env python
"""
Test script to verify YOLOv5-Face model loading on CPU
"""

import os
import sys

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'yolov5-face'))

import torch
import numpy as np
import cv2

print("=" * 60)
print("YOLOv5-Face Model Loading Test")
print("=" * 60)

# Check PyTorch and CUDA
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Set device
device = torch.device('cpu')
print(f"\nUsing device: {device}")

# Model path
model_path = os.path.join(script_dir, 'models', 'yolov5l6_best.pt')
if not os.path.exists(model_path):
    # Try yolov5-face weights directory
    model_path = os.path.join(script_dir, 'yolov5-face', 'weights', 'yolov5l6_best.pt')

print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

if not os.path.exists(model_path):
    print("\nERROR: Model file not found!")
    sys.exit(1)

# Try to import yolov5-face modules
print("\n" + "-" * 60)
print("Testing yolov5-face module imports...")
print("-" * 60)

try:
    import yaml
    print("✓ PyYAML imported successfully")
except ImportError as e:
    print(f"✗ PyYAML import failed: {e}")
    print("  Please install: pip install PyYAML")
    sys.exit(1)

try:
    from models.experimental import attempt_load
    print("✓ attempt_load imported successfully")
except ImportError as e:
    print(f"✗ attempt_load import failed: {e}")
    sys.exit(1)

try:
    from utils.datasets import letterbox
    print("✓ letterbox imported successfully")
except ImportError as e:
    print(f"✗ letterbox import failed: {e}")
    sys.exit(1)

try:
    from utils.general import non_max_suppression_face, scale_coords
    print("✓ NMS and scale_coords imported successfully")
except ImportError as e:
    print(f"✗ General utils import failed: {e}")
    sys.exit(1)

# Import our scale_coords_landmarks
from face_detection.face_detector import scale_coords_landmarks
print("✓ scale_coords_landmarks imported successfully")

# Load model
print("\n" + "-" * 60)
print("Loading model...")
print("-" * 60)

try:
    print(f"Loading model from: {model_path}")
    print(f"Mapping to device: {device}")
    
    model = attempt_load(model_path, map_location=device)
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully!")
    print(f"  Model type: {type(model)}")
    print(f"  Model device: {next(model.parameters()).device}")
    
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test inference with dummy image
print("\n" + "-" * 60)
print("Testing inference with dummy image...")
print("-" * 60)

try:
    # Create a dummy image (640x640 RGB)
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Preprocess
    img = letterbox(dummy_img, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    print(f"  Input shape: {img.shape}")
    
    # Inference
    with torch.no_grad():
        pred = model(img)[0]
    
    print(f"  Output shape: {pred.shape}")
    print("✓ Inference successful!")
    
except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test NMS
print("\n" + "-" * 60)
print("Testing NMS...")
print("-" * 60)

try:
    conf_thres = 0.2
    iou_thres = 0.5
    
    pred_nms = non_max_suppression_face(pred, conf_thres, iou_thres)
    
    print(f"  Detections after NMS: {len(pred_nms[0])}")
    print("✓ NMS successful!")
    
except Exception as e:
    print(f"✗ NMS failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test FaceDetector class
print("\n" + "-" * 60)
print("Testing FaceDetector class...")
print("-" * 60)

try:
    from face_detection.face_detector import FaceDetector, create_face_detector
    
    detector = create_face_detector(
        model_path=model_path,
        device='cpu',
        align_size=224
    )
    
    print("✓ FaceDetector created successfully!")
    
    # Test detection
    aligned, original, info = detector.detect_and_align(dummy_img)
    
    if aligned is not None:
        print(f"  Aligned face shape: {aligned.shape}")
        print(f"  Detection info: {info}")
        print("✓ Face detection successful!")
    else:
        print("  No face detected (expected for random image)")
        print("✓ FaceDetector works correctly!")
    
except Exception as e:
    print(f"✗ FaceDetector test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nThe YOLOv5-Face model can be loaded and run on CPU.")
