#!/usr/bin/env python
"""
Minimal test script for Face Match Service on CPU
Tests only the core components without full service
"""

import os
import sys

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'yolov5-face'))

print("=" * 70)
print("Face Match Service - Minimal Test on CPU")
print("=" * 70)

# Check PyTorch
print(f"\n[1] PyTorch Check")
print("-" * 70)
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    device = torch.device('cpu')
    print(f"  ✓ Using device: {device}")
except ImportError as e:
    print(f"  ✗ PyTorch not installed: {e}")
    sys.exit(1)

# Check model files
print(f"\n[2] Model Files Check")
print("-" * 70)
model_paths = [
    ("Face Detector", "./yolov5-face/weights/yolov5l6_best.pt"),
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

# Core dependencies
core_deps = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("PIL", "Pillow"),
    ("yaml", "PyYAML"),
    ("thop", "thop"),
]

missing_core = []
for module, package in core_deps:
    try:
        __import__(module)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (MISSING)")
        missing_core.append(package)

# Cross-modal dependencies
cross_deps = [
    ("clip", "clip (git+https://github.com/openai/CLIP.git)"),
    ("facenet_pytorch", "facenet-pytorch"),
]

missing_cross = []
for module, package in cross_deps:
    try:
        __import__(module)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (MISSING)")
        missing_cross.append(package)

# Test Face Detection
print(f"\n[4] Face Detection Test")
print("-" * 70)

if missing_core:
    print(f"  ✗ Skipping - missing core dependencies: {', '.join(missing_core)}")
else:
    try:
        from models.experimental import attempt_load
        from utils.datasets import letterbox
        from utils.general import non_max_suppression_face, scale_coords
        print("  ✓ YOLOv5-face modules imported")
        
        # Load model
        model_path = "./yolov5-face/weights/yolov5l6_best.pt"
        if os.path.exists(model_path):
            model = attempt_load(model_path, map_location=device)
            model.to(device)
            model.eval()
            print("  ✓ Face detector model loaded")
            
            # Test inference
            import numpy as np
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img = letterbox(dummy_img, new_shape=640)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device).float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            with torch.no_grad():
                pred = model(img)[0]
            
            pred_nms = non_max_suppression_face(pred, 0.2, 0.5)
            print(f"  ✓ Inference successful (detections: {len(pred_nms[0])})")
        else:
            print("  ✗ Model file not found")
            
    except Exception as e:
        print(f"  ✗ Face detection failed: {e}")
        import traceback
        traceback.print_exc()

# Test Cross-Modal Matcher
print(f"\n[5] Cross-Modal Matcher Test")
print("-" * 70)

if missing_cross:
    print(f"  ✗ Skipping - missing dependencies: {', '.join(missing_cross)}")
    print(f"  Install with: pip install {' '.join(missing_cross)}")
else:
    try:
        import clip
        from facenet_pytorch import InceptionResnetV1
        print("  ✓ CLIP and FaceNet imported")
        
        # Test CLIP
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print("  ✓ CLIP model loaded")
        
        # Test FaceNet
        face_encoder = InceptionResnetV1(pretrained='casia-webface', classify=False, device=device)
        print("  ✓ FaceNet model loaded")
        
        # Test feature extraction
        from cross_modal.matcher import CrossModalFaceMatcher, ImagePreprocessor, create_matcher
        
        matcher = create_matcher(device='cpu', embedding_dim=512, weights_path=None)
        preprocessor = ImagePreprocessor(device='cpu')
        print("  ✓ Cross-modal matcher created")
        
        # Test with dummy image
        import numpy as np
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        fn_tensor, clip_tensor = preprocessor.preprocess(dummy_img)
        
        with torch.no_grad():
            embedding = matcher.get_embeddings(fn_tensor, clip_tensor)
        
        print(f"  ✓ Feature extraction successful (embedding shape: {embedding.shape})")
        
    except Exception as e:
        print(f"  ✗ Cross-modal matcher failed: {e}")
        import traceback
        traceback.print_exc()

# Summary
print(f"\n" + "=" * 70)
print("Summary")
print("=" * 70)

if missing_core:
    print(f"\nMissing core dependencies: {', '.join(missing_core)}")
    print(f"Install with: pip install {' '.join(missing_core)}")

if missing_cross:
    print(f"\nMissing cross-modal dependencies: {', '.join(missing_cross)}")
    print(f"Install with: pip install {' '.join(missing_cross)}")

if not missing_core and not missing_cross:
    print("\n✓ All tests passed!")
else:
    print("\n✗ Some tests failed due to missing dependencies")
