"""
Face Detection Module
Based on YOLOv5-Face from AnyFace-face-detect
Uses the original yolov5-face modules for model loading
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import copy
import warnings
import sys
import os
import math
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from copy import deepcopy

# Try to import skimage for face alignment
try:
    from skimage import transform as trans
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[Warning] scikit-image not available. Face alignment may not work. Install with: pip install scikit-image")

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

# ============================================================================
# Add yolov5-face to path
# ============================================================================

YOLOV5_FACE_PATH = os.path.join(os.path.dirname(__file__), '..', 'yolov5-face')
if os.path.exists(YOLOV5_FACE_PATH):
    sys.path.insert(0, YOLOV5_FACE_PATH)

# Try to import from yolov5-face
try:
    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import non_max_suppression_face, scale_coords
    # scale_coords_landmarks is not in utils.general, we define it below
    YOLOV5_AVAILABLE = True
    print("[FaceDetector] Successfully imported yolov5-face modules")
except ImportError as e:
    YOLOV5_AVAILABLE = False
    print(f"[Warning] Could not import from yolov5-face: {e}")
    print("[Warning] Make sure the yolov5-face directory exists in python_service/")


# ============================================================================
# scale_coords_landmarks function (not in original utils.general)
# ============================================================================

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale landmark coordinates from img1_shape to img0_shape.
    This function is not in the original utils.general, so we define it here.
    
    Args:
        img1_shape: Shape of the resized image (h, w)
        coords: Landmark coordinates tensor (N, 10) - 5 landmarks x 2 coords
        img0_shape: Shape of the original image (h, w)
        ratio_pad: Optional ratio and padding from letterbox
    
    Returns:
        Rescaled coordinates
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip coords
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


# ============================================================================
# Face Alignment Functions
# ============================================================================

# Reference landmarks for face alignment
SRC1 = np.array([
    [51.642, 50.115],
    [57.617, 49.990],
    [35.740, 69.007],
    [51.157, 89.050],
    [57.025, 89.702]
], dtype=np.float32)

SRC2 = np.array([
    [45.031, 50.118],
    [65.568, 50.872],
    [39.677, 68.111],
    [45.177, 86.190],
    [64.246, 86.758]
], dtype=np.float32)

SRC3 = np.array([
    [39.730, 51.138],
    [72.270, 51.138],
    [56.000, 68.493],
    [42.463, 87.010],
    [69.537, 87.010]
], dtype=np.float32)

SRC4 = np.array([
    [46.845, 50.872],
    [67.382, 50.118],
    [72.737, 68.111],
    [48.167, 86.758],
    [67.236, 86.190]
], dtype=np.float32)

SRC5 = np.array([
    [54.796, 49.990],
    [60.771, 50.115],
    [76.673, 69.007],
    [55.388, 89.702],
    [61.257, 89.050]
], dtype=np.float32)

SRC = np.array([SRC1, SRC2, SRC3, SRC4, SRC5])
SRC_MAP = {112: SRC, 224: SRC * 2}

ARCFACE_SRC = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

ARCFACE_SRC = np.expand_dims(ARCFACE_SRC, axis=0)


def estimate_norm(lmk: np.ndarray, image_size: int = 112, mode: str = 'arcface') -> Tuple[np.ndarray, int]:
    """
    Estimate transformation matrix for face alignment.
    
    Args:
        lmk: 5 facial landmarks (5, 2)
        image_size: Output image size
        mode: Alignment mode ('arcface' or other)
    
    Returns:
        Transformation matrix and pose index
    """
    if not SKIMAGE_AVAILABLE:
        raise RuntimeError("scikit-image is required for face alignment")
    
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    
    if mode == 'arcface':
        assert image_size == 112
        src = ARCFACE_SRC
    else:
        src = SRC_MAP[image_size]
    
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    
    return min_M, min_index


def norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = 112, mode: str = 'arcface') -> np.ndarray:
    """
    Align and crop face based on landmarks.
    
    Args:
        img: Input image
        landmark: 5 facial landmarks
        image_size: Output image size
        mode: Alignment mode
    
    Returns:
        Aligned face image
    """
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


# ============================================================================
# Face Detector Class
# ============================================================================

class FaceDetector:
    """
    Face Detector using YOLOv5-Face.
    Detects faces and landmarks, then aligns faces to specified size.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        conf_threshold: float = 0.2,
        iou_threshold: float = 0.5,
        align_size: int = 224
    ):
        """
        Initialize Face Detector.
        
        Args:
            model_path: Path to YOLOv5-Face weights
            device: Device to run inference ('cuda' or 'cpu')
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
            align_size: Output size for aligned face (default 224x224)
        """
        # Properly detect device
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"[FaceDetector] CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.align_size = align_size
        
        if not YOLOV5_AVAILABLE:
            raise RuntimeError(
                "yolov5-face modules not available. "
                "Make sure the yolov5-face directory is in python_service/"
            )
        
        # Load model using original attempt_load
        self.model = self._load_model(model_path)
        print(f"[FaceDetector] Model loaded on {self.device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load YOLOv5-Face model using original attempt_load."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Use original attempt_load from yolov5-face
            model = attempt_load(model_path, map_location=self.device)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def detect_and_align(
        self,
        image: np.ndarray,
        return_largest: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """
        Detect faces in image and align the largest face.
        
        Args:
            image: Input image (BGR format)
            return_largest: Whether to return only the largest face
        
        Returns:
            aligned_face: Aligned face image (224x224 by default)
            original_face: Original cropped face
            info: Detection info including bbox, landmarks, confidence
        """
        # Prepare image
        img0 = copy.deepcopy(image)
        
        # Use original letterbox function
        img, ratio, pad = letterbox(img0, new_shape=640)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        
        # Convert to tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred = self.model(img)[0]
        
        # Use original NMS
        pred = non_max_suppression_face(pred, self.conf_threshold, self.iou_threshold)
        
        if len(pred[0]) == 0:
            return None, None, None
        
        # Process detections
        det = pred[0]
        
        # Scale coordinates using original functions
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], img0.shape).round()
        
        # Find largest face
        if return_largest:
            areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            max_idx = areas.argmax()
            det = det[max_idx:max_idx + 1]
        
        # Get detection info
        x1, y1, x2, y2 = map(int, det[0, :4])
        conf = float(det[0, 4])
        landmarks = det[0, 5:15].cpu().numpy().reshape(5, 2)
        
        # Align face
        if SKIMAGE_AVAILABLE:
            aligned_face = norm_crop(img0, landmarks, image_size=112, mode='arcface')
            # Resize to requested size
            if self.align_size != 112:
                aligned_face = cv2.resize(aligned_face, (self.align_size, self.align_size))
        else:
            # Fallback: just crop and resize
            aligned_face = img0[y1:y2, x1:x2]
            aligned_face = cv2.resize(aligned_face, (self.align_size, self.align_size))
        
        # Also get original cropped face
        original_face = img0[y1:y2, x1:x2]
        
        info = {
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'landmarks': landmarks.tolist()
        }
        
        return aligned_face, original_face, info


# ============================================================================
# Convenience Functions
# ============================================================================

def create_face_detector(
    model_path: str,
    device: str = 'cuda',
    align_size: int = 224
) -> FaceDetector:
    """
    Create a FaceDetector instance.
    
    Args:
        model_path: Path to YOLOv5-Face weights
        device: Device to run inference
        align_size: Output size for aligned face
    
    Returns:
        FaceDetector instance
    """
    return FaceDetector(
        model_path=model_path,
        device=device,
        align_size=align_size
    )
