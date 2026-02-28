"""
Face Detection Module
Based on YOLOv5-Face from AnyFace-face-detect
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import copy
import warnings
import sys
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from skimage import transform as trans

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

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
# YOLOv5-Face Model Components
# ============================================================================

def autopad(k, p=None):
    """Pad to 'same' shape outputs."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation."""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""
    
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""
    
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    
    def forward(self, x):
        return torch.cat(x, self.d)


class Detect(nn.Module):
    """YOLOv5 Detect layer for face detection with landmarks."""
    
    stride = None
    export_cat = False
    
    def __init__(self, nc=1, anchors=(), ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + 10  # number of outputs per anchor (class + bbox + conf + 5 landmarks)
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
    
    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                
                y = torch.full_like(x[i], 0)
                class_range = list(range(5)) + list(range(15, 15 + self.nc))
                y[..., class_range] = x[i][..., class_range].sigmoid()
                y[..., 5:15] = x[i][..., 5:15]
                
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                
                y[..., 5:7] = y[..., 5:7] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                y[..., 7:9] = y[..., 7:9] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                y[..., 9:11] = y[..., 9:11] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                y[..., 11:13] = y[..., 11:13] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                y[..., 13:15] = y[..., 13:15] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                
                z.append(y.view(bs, -1, self.no))
        
        return x if self.training else (torch.cat(z, 1), x)
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


# ============================================================================
# YOLOv5-Face Model Builder
# ============================================================================

class YOLOv5Face(nn.Module):
    """YOLOv5-Face model for face detection with landmarks."""
    
    def __init__(self, model_dict: dict, ch: int = 3):
        super().__init__()
        self.yaml = model_dict
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        self.model, self.save = self._parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        
        # Build strides, anchors
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 128
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()
        
        # Initialize weights
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x
    
    def _initialize_biases(self):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99))
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)
    
    def _parse_model(self, d, ch):
        anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        no = na * (nc + 5)
        
        layers, save, c2 = [], [], ch[-1]
        
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
            m = eval(m) if isinstance(m, str) else m
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a
                except:
                    pass
            
            n = max(round(n * gd), 1) if n > 1 else n
            
            if m in [Conv, Bottleneck, C3, SPPF]:
                c1, c2 = ch[f], args[0]
                c2 = int(c2 * gw) if c2 != no else c2
                args = [c1, c2, *args[1:]]
                if m in [C3]:
                    args.insert(2, n)
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
            elif m is Detect:
                args.append([ch[x + 1] for x in f])
                if isinstance(args[1], int):
                    args[1] = [list(range(args[1] * 2))] * len(f)
            else:
                c2 = ch[f]
            
            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
            m_.i, m_.f = i, f
            layers.append(m_)
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            ch.append(c2)
        
        return nn.Sequential(*layers), sorted(save)


import math
from copy import deepcopy


def make_divisible(x, divisor):
    """Make divisible."""
    return math.ceil(x / divisor) * divisor


# ============================================================================
# YOLOv5l6-Face Model Configuration
# ============================================================================

YOLOV5L6_CONFIG = {
    'nc': 1,  # number of classes
    'depth_multiple': 1.0,
    'width_multiple': 1.0,
    'anchors': [[19, 27], [44, 40], [38, 94], [96, 68], [86, 152], [180, 137]],
    'ch': 3,
    'backbone': [
        [-1, 1, Conv, [64, 6, 2, 2]],
        [-1, 1, Conv, [128, 3, 2]],
        [-1, 3, C3, [128]],
        [-1, 1, Conv, [256, 3, 2]],
        [-1, 6, C3, [256]],
        [-1, 1, Conv, [512, 3, 2]],
        [-1, 9, C3, [512]],
        [-1, 1, Conv, [768, 3, 2]],
        [-1, 3, C3, [768]],
        [-1, 1, SPPF, [1024, 5]],
    ],
    'head': [
        [-1, 1, Conv, [768, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 8], 1, Concat, [1]],
        [-1, 3, C3, [768, False]],
        [-1, 1, Conv, [512, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 6], 1, Concat, [1]],
        [-1, 3, C3, [512, False]],
        [-1, 1, Conv, [256, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 4], 1, Concat, [1]],
        [-1, 3, C3, [256, False]],
        [-1, 1, Conv, [256, 3, 2]],
        [[-1, 16], 1, Concat, [1]],
        [-1, 3, C3, [512, False]],
        [-1, 1, Conv, [512, 3, 2]],
        [[-1, 13], 1, Concat, [1]],
        [-1, 3, C3, [768, False]],
        [-1, 1, Conv, [768, 3, 2]],
        [[-1, 10], 1, Concat, [1]],
        [-1, 3, C3, [1024, False]],
        [[17, 20, 23], 1, Detect, [1, 'anchors']],
    ]
}


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
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.align_size = align_size
        
        # Load model
        self.model = self._load_model(model_path)
        print(f"[FaceDetector] Model loaded on {self.device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load YOLOv5-Face model."""
        # Create model from config
        model = YOLOv5Face(YOLOV5L6_CONFIG, ch=3)
        
        # Load weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            # Handle different state_dict formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Remove 'model.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=False)
        else:
            print(f"[FaceDetector] Warning: Model file not found at {model_path}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def letterbox(self, img: np.ndarray, new_shape: int = 640) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Resize image with aspect ratio preserved.
        
        Args:
            img: Input image
            new_shape: Target size
        
        Returns:
            Resized image, scale ratio, padding
        """
        shape = img.shape[:2]
        r = new_shape / max(shape)
        
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return img, r, (dw, dh)
    
    def non_max_suppression_face(self, prediction: torch.Tensor) -> List[torch.Tensor]:
        """
        Non-Maximum Suppression for face detection.
        
        Args:
            prediction: Model output predictions
        
        Returns:
            List of detections after NMS
        """
        bs = prediction.shape[0]
        nc = 1
        output = [torch.zeros((0, 16), device=prediction.device)] * bs
        
        for xi, x in enumerate(prediction):
            # x shape: (num_anchors, 16) - [x, y, w, h, conf, lmk1_x, lmk1_y, ..., lmk5_x, lmk5_y, class]
            
            # Filter by confidence
            x = x.T
            box = x[:4].T  # x, y, w, h
            conf = x[4:5].T  # confidence
            landmarks = x[5:15].T  # 5 landmarks
            class_conf = x[15:16].T  # class confidence
            
            # Filter by confidence threshold
            conf_mask = conf.squeeze() > self.conf_threshold
            x = torch.cat((box, conf, landmarks, class_conf), dim=1)[conf_mask]
            
            if not x.shape[0]:
                continue
            
            # NMS
            boxes = x[:, :4]
            scores = x[:, 4] * x[:, 15]
            
            try:
                import torchvision
                keep = torchvision.ops.nms(boxes, scores, self.iou_threshold)
            except:
                # Fallback NMS
                keep = []
                order = scores.argsort(descending=True)
                while len(order) > 0:
                    idx = order[0].item()
                    keep.append(idx)
                    if len(order) == 1:
                        break
                    iou = self._compute_iou(boxes[idx], boxes[order[1:]])
                    order = order[1:][iou < self.iou_threshold]
                keep = torch.tensor(keep, device=prediction.device)
            
            output[xi] = x[keep]
        
        return output
    
    def _compute_iou(self, box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU between one box and multiple boxes."""
        # box: [x, y, w, h]
        # boxes: [N, 4]
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        x2 = box[0] + box[2] / 2
        y2 = box[1] + box[3] / 2
        
        xx1 = boxes[:, 0] - boxes[:, 2] / 2
        yy1 = boxes[:, 1] - boxes[:, 3] / 2
        xx2 = boxes[:, 0] + boxes[:, 2] / 2
        yy2 = boxes[:, 1] + boxes[:, 3] / 2
        
        inter_x1 = torch.max(x1, xx1)
        inter_y1 = torch.max(y1, yy1)
        inter_x2 = torch.min(x2, xx2)
        inter_y2 = torch.min(y2, yy2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        """Scale landmark coordinates from resized image to original image."""
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        else:
            gain = ratio_pad[0]
            pad = ratio_pad[1]
        
        coords[:, [0, 2, 4, 6, 8]] -= pad[0]
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]
        coords[:, :10] /= gain
        
        coords[:, 0].clamp_(0, img0_shape[1])
        coords[:, 1].clamp_(0, img0_shape[0])
        coords[:, 2].clamp_(0, img0_shape[1])
        coords[:, 3].clamp_(0, img0_shape[0])
        coords[:, 4].clamp_(0, img0_shape[1])
        coords[:, 5].clamp_(0, img0_shape[0])
        coords[:, 6].clamp_(0, img0_shape[1])
        coords[:, 7].clamp_(0, img0_shape[0])
        coords[:, 8].clamp_(0, img0_shape[1])
        coords[:, 9].clamp_(0, img0_shape[0])
        
        return coords
    
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
        h0, w0 = image.shape[:2]
        
        # Letterbox resize
        img, ratio, pad = self.letterbox(image, new_shape=640)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        
        # Convert to tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred = self.model(img)
            if isinstance(pred, tuple):
                pred = pred[0]
        
        # NMS
        pred = self.non_max_suppression_face(pred)
        
        if len(pred[0]) == 0:
            return None, None, None
        
        # Process detections
        det = pred[0]
        
        # Scale coordinates
        det[:, :4] = self._scale_coords(img.shape[2:], det[:, :4], img0.shape)
        det[:, 5:15] = self.scale_coords_landmarks(img.shape[2:], det[:, 5:15], img0.shape)
        
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
        aligned_face = norm_crop(img0, landmarks, image_size=self.align_size, mode='arcface')
        
        # Resize to 224x224 if needed
        if self.align_size != 224:
            aligned_face = cv2.resize(aligned_face, (224, 224))
        
        # Also get original cropped face
        original_face = img0[y1:y2, x1:x2]
        
        info = {
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'landmarks': landmarks.tolist()
        }
        
        return aligned_face, original_face, info
    
    def _scale_coords(self, img1_shape, coords, img0_shape):
        """Scale bounding box coordinates."""
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        
        coords[:, [0, 2]].clamp_(0, img0_shape[1])
        coords[:, [1, 3]].clamp_(0, img0_shape[0])
        
        return coords


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
