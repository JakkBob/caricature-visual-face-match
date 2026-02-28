#!/usr/bin/env python
"""
Complete test script for Face Match Service on CPU
Tests all components: face detection, feature extraction, cross-modal matching
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

print("=" * 70)
print("Face Match Service - Complete Test on CPU")
print("=" * 70)

# Check PyTorch and CUDA
print(f"\n[1] Environment Check")
print("-" * 70)
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
device = torch.device('cpu')
print(f"  Using device: {device}")

# Check model files
print(f"\n[2] Model Files Check")
print("-" * 70)
model_paths = [
    ("Face Detector", "./models/yolov5l6_best.pt"),
    ("Face Detector (yolov5-face)", "./yolov5-face/weights/yolov5l6_best.pt"),
    ("Cross-Modal Matcher", "./models/cross_modal_matcher.pt"),
]

for name, path in model_paths:
    exists = os.path.exists(path)
    size = os.path.getsize(path) / (1024*1024) if exists else 0
    status = "✓" if exists else "✗"
    print(f"  {status} {name}: {path} ({size:.1f} MB)" if exists else f"  {status} {name}: {path} (NOT FOUND)")

# Check dependencies
print(f"\n[3] Dependencies Check")
print("-" * 70)

dependencies = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("PIL", "Pillow"),
    ("yaml", "PyYAML"),
    ("clip", "clip (git+https://github.com/openai/CLIP.git)"),
    ("facenet_pytorch", "facenet-pytorch"),
    ("torchvision", "torchvision"),
    ("skimage", "scikit-image"),
    ("thop", "thop"),
]

missing_deps = []
for module, package in dependencies:
    try:
        __import__(module)
        print(f"  ✓ {package}")
    except ImportError as e:
        print(f"  ✗ {package}: {e}")
        missing_deps.append(package)

if missing_deps:
    print(f"\n  Missing dependencies: {', '.join(missing_deps)}")
    print(f"  Install with: pip install {' '.join(missing_deps)}")

# Test Face Detection
print(f"\n[4] Face Detection Test")
print("-" * 70)

try:
    # Find model path
    model_path = "./models/yolov5l6_best.pt"
    if not os.path.exists(model_path):
        model_path = "./yolov5-face/weights/yolov5l6_best.pt"
    
    if not os.path.exists(model_path):
        print("  ✗ Model file not found!")
    else:
        from face_detection.face_detector import FaceDetector, create_face_detector
        
        detector = create_face_detector(
            model_path=model_path,
            device='cpu',
            align_size=224
        )
        print("  ✓ Face detector created successfully")
        
        # Test with dummy image
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        aligned, original, info = detector.detect_and_align(dummy_img)
        
        if aligned is not None:
            print(f"  ✓ Detection works (aligned shape: {aligned.shape})")
        else:
            print("  ✓ Detection works (no face in random image - expected)")
        
except Exception as e:
    print(f"  ✗ Face detection failed: {e}")
    import traceback
    traceback.print_exc()

# Test Cross-Modal Matcher
print(f"\n[5] Cross-Modal Matcher Test")
print("-" * 70)

try:
    from cross_modal.matcher import CrossModalFaceMatcher, ImagePreprocessor, create_matcher
    
    # Check if CLIP is available
    try:
        import clip
        print("  ✓ CLIP is available")
    except ImportError:
        print("  ✗ CLIP not available - install with: pip install git+https://github.com/openai/CLIP.git")
        raise
    
    # Check if facenet-pytorch is available
    try:
        from facenet_pytorch import InceptionResnetV1
        print("  ✓ facenet-pytorch is available")
    except ImportError:
        print("  ✗ facenet-pytorch not available - install with: pip install facenet-pytorch")
        raise
    
    # Create matcher
    matcher = create_matcher(
        device='cpu',
        embedding_dim=512,
        weights_path=None
    )
    print("  ✓ Cross-modal matcher created successfully")
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(device='cpu')
    print("  ✓ Image preprocessor created successfully")
    
    # Test feature extraction
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    fn_tensor, clip_tensor = preprocessor.preprocess(dummy_img)
    print(f"  ✓ Preprocessing works (FaceNet: {fn_tensor.shape}, CLIP: {clip_tensor.shape})")
    
    # Extract features
    with torch.no_grad():
        embedding = matcher.get_embeddings(fn_tensor, clip_tensor)
    print(f"  ✓ Feature extraction works (embedding shape: {embedding.shape})")
    
except Exception as e:
    print(f"  ✗ Cross-modal matcher failed: {e}")
    import traceback
    traceback.print_exc()

# Test Full Service
print(f"\n[6] Full Service Test")
print("-" * 70)

try:
    from main import FaceMatchService, Config
    
    # Override config for testing
    config = Config()
    config.DEVICE = 'cpu'
    
    # Find model path
    if not os.path.exists(config.FACE_DETECTOR_MODEL):
        config.FACE_DETECTOR_MODEL = "./yolov5-face/weights/yolov5l6_best.pt"
    
    service = FaceMatchService(config)
    print(f"  Service device: {service.device}")
    
    # Initialize
    service.initialize()
    
    print(f"  Models loaded: {service.models_loaded}")
    
    if service.models_loaded['face_detector']:
        print("  ✓ Face detector loaded")
    else:
        print("  ✗ Face detector not loaded")
    
    if service.models_loaded['cross_modal_matcher']:
        print("  ✓ Cross-modal matcher loaded")
    else:
        print("  ✗ Cross-modal matcher not loaded")
    
except Exception as e:
    print(f"  ✗ Full service test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
