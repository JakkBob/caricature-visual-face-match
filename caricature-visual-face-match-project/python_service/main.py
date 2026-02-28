"""
Python Backend Service for Face Matching
FastAPI-based service for face detection and cross-modal matching
"""

import os
import sys
import io
import base64
import json
import time
import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import cv2
from PIL import Image
import torch

# FastAPI
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from face_detection.face_detector import FaceDetector, create_face_detector
from cross_modal.matcher import CrossModalFaceMatcher, ImagePreprocessor, create_matcher

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Service configuration."""
    
    # Model paths - use yolov5-face/weights directory for face detector
    FACE_DETECTOR_MODEL = os.environ.get(
        'FACE_DETECTOR_MODEL',
        './yolov5-face/weights/yolov5l6_best.pt'
    )
    CROSS_MODAL_MODEL = os.environ.get(
        'CROSS_MODAL_MODEL',
        './models/cross_modal_matcher.pt'
    )
    
    # Device - properly detect CUDA availability
    _env_device = os.environ.get('DEVICE', '')
    if _env_device:
        DEVICE = _env_device
    else:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Face detection settings
    FACE_ALIGN_SIZE = 224
    CONFIDENCE_THRESHOLD = 0.2
    IOU_THRESHOLD = 0.5
    
    # Matching settings
    EMBEDDING_DIM = 512
    TOP_K = 10
    MATCH_THRESHOLD = 0.5


# ============================================================================
# Request/Response Models
# ============================================================================

class DetectRequest(BaseModel):
    """Face detection request."""
    image: str  # Base64 encoded image
    align_size: Optional[int] = 224


class DetectResponse(BaseModel):
    """Face detection response."""
    success: bool
    message: Optional[str] = None
    faces: Optional[List[Dict[str, Any]]] = None
    aligned_image: Optional[str] = None  # Base64 encoded aligned face


class ExtractRequest(BaseModel):
    """Feature extraction request."""
    image: str  # Base64 encoded image
    modality: str  # 'face' or 'caricature'


class ExtractResponse(BaseModel):
    """Feature extraction response."""
    success: bool
    message: Optional[str] = None
    feature: Optional[List[float]] = None
    dimension: Optional[int] = None


class MatchRequest(BaseModel):
    """Matching request."""
    query_image: str  # Base64 encoded image
    query_modality: Optional[str] = 'face'  # 'face' or 'caricature'
    gallery_images: List[str]  # List of base64 encoded images
    gallery_ids: Optional[List[str]] = None
    top_k: Optional[int] = 10


class MatchResponse(BaseModel):
    """Matching response."""
    success: bool
    message: Optional[str] = None
    matches: Optional[List[Dict[str, Any]]] = None
    query_feature: Optional[List[float]] = None


class SimilarityRequest(BaseModel):
    """Similarity computation request."""
    image1: str  # Base64 encoded image
    image2: str  # Base64 encoded image


class SimilarityResponse(BaseModel):
    """Similarity computation response."""
    success: bool
    message: Optional[str] = None
    similarity: Optional[float] = None


class StatusResponse(BaseModel):
    """Service status response."""
    status: str
    device: str
    models_loaded: Dict[str, bool]
    gpu_available: bool


# ============================================================================
# Service Class
# ============================================================================

class FaceMatchService:
    """Face Matching Service."""
    
    def __init__(self, config: Config = Config()):
        """Initialize service."""
        self.config = config
        
        # Properly detect device - check CUDA availability again
        if config.DEVICE == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'
        else:
            self.device = config.DEVICE
        
        # Models
        self.face_detector: Optional[FaceDetector] = None
        self.matcher: Optional[CrossModalFaceMatcher] = None
        self.preprocessor: Optional[ImagePreprocessor] = None
        
        # Status
        self.models_loaded = {
            'face_detector': False,
            'cross_modal_matcher': False
        }
    
    def initialize(self):
        """Initialize models."""
        logger.info(f"Initializing Face Match Service on device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Initialize face detector
        try:
            if os.path.exists(self.config.FACE_DETECTOR_MODEL):
                self.face_detector = create_face_detector(
                    model_path=self.config.FACE_DETECTOR_MODEL,
                    device=self.device,
                    align_size=self.config.FACE_ALIGN_SIZE
                )
                self.models_loaded['face_detector'] = True
                logger.info("Face detector loaded successfully")
            else:
                logger.warning(f"Face detector model not found at {self.config.FACE_DETECTOR_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load face detector: {e}")
        
        # Initialize cross-modal matcher
        try:
            weights_path = None
            if os.path.exists(self.config.CROSS_MODAL_MODEL):
                weights_path = self.config.CROSS_MODAL_MODEL
            
            self.matcher = create_matcher(
                device=self.device,
                embedding_dim=self.config.EMBEDDING_DIM,
                weights_path=weights_path
            )
            self.preprocessor = ImagePreprocessor(device=self.device)
            self.models_loaded['cross_modal_matcher'] = True
            logger.info("Cross-modal matcher loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-modal matcher: {e}")
        
        logger.info("Service initialization complete")
    
    def decode_image(self, base64_image: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        # Remove data URL prefix if present
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        
        image_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    
    def encode_image(self, image: np.ndarray) -> str:
        """Encode numpy array to base64 image."""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode('utf-8')
    
    def detect_faces(self, image: np.ndarray, align_size: int = 224) -> Dict[str, Any]:
        """Detect and align faces in image."""
        if self.face_detector is None:
            return {
                'success': False,
                'message': 'Face detector not loaded'
            }
        
        try:
            aligned_face, original_face, info = self.face_detector.detect_and_align(
                image,
                return_largest=True
            )
            
            if aligned_face is None:
                return {
                    'success': False,
                    'message': 'No face detected'
                }
            
            # Resize to requested size
            if align_size != 224:
                aligned_face = cv2.resize(aligned_face, (align_size, align_size))
            
            return {
                'success': True,
                'faces': [info] if info else [],
                'aligned_image': self.encode_image(aligned_face)
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Detection failed: {str(e)}'
            }
    
    def extract_features(
        self,
        image: np.ndarray,
        modality: str = 'face'
    ) -> Dict[str, Any]:
        """Extract features from image."""
        if self.matcher is None or self.preprocessor is None:
            return {
                'success': False,
                'message': 'Feature extractor not loaded'
            }
        
        try:
            # Preprocess image
            facenet_tensor, clip_tensor = self.preprocessor.preprocess(image)
            
            # Extract features
            with torch.no_grad():
                embedding = self.matcher.get_embeddings(facenet_tensor, clip_tensor)
            
            feature = embedding.cpu().numpy().flatten().tolist()
            
            return {
                'success': True,
                'feature': feature,
                'dimension': len(feature)
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Feature extraction failed: {str(e)}'
            }
    
    def match(
        self,
        query_image: np.ndarray,
        gallery_images: List[np.ndarray],
        gallery_ids: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Match query image against gallery."""
        if self.matcher is None or self.preprocessor is None:
            return {
                'success': False,
                'message': 'Matcher not loaded'
            }
        
        try:
            # Extract query features
            query_fn, query_clip = self.preprocessor.preprocess(query_image)
            
            with torch.no_grad():
                query_embedding = self.matcher.get_embeddings(query_fn, query_clip)
            
            # Extract gallery features
            gallery_embeddings = []
            for img in gallery_images:
                fn_tensor, clip_tensor = self.preprocessor.preprocess(img)
                with torch.no_grad():
                    emb = self.matcher.get_embeddings(fn_tensor, clip_tensor)
                gallery_embeddings.append(emb)
            
            # Stack gallery embeddings
            gallery_stack = torch.cat(gallery_embeddings, dim=0)
            
            # Compute similarities
            similarities = self.matcher.compute_similarity(query_embedding, gallery_stack)
            similarities = similarities.cpu().numpy().flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Build results
            matches = []
            for rank, idx in enumerate(top_indices):
                match = {
                    'index': int(idx),
                    'similarity': float(similarities[idx]),
                    'rank': rank + 1,
                    'is_match': similarities[idx] >= self.config.MATCH_THRESHOLD
                }
                if gallery_ids:
                    match['id'] = gallery_ids[idx]
                matches.append(match)
            
            return {
                'success': True,
                'matches': matches,
                'query_feature': query_embedding.cpu().numpy().flatten().tolist()
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Matching failed: {str(e)}'
            }
    
    def compute_similarity(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Dict[str, Any]:
        """Compute similarity between two images."""
        if self.matcher is None or self.preprocessor is None:
            return {
                'success': False,
                'message': 'Matcher not loaded'
            }
        
        try:
            # Extract features
            fn1, clip1 = self.preprocessor.preprocess(image1)
            fn2, clip2 = self.preprocessor.preprocess(image2)
            
            with torch.no_grad():
                emb1 = self.matcher.get_embeddings(fn1, clip1)
                emb2 = self.matcher.get_embeddings(fn2, clip2)
            
            # Compute similarity
            similarity = self.matcher.compute_similarity(emb1, emb2)
            similarity_value = float(similarity.cpu().numpy().flatten()[0])
            
            return {
                'success': True,
                'similarity': similarity_value
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Similarity computation failed: {str(e)}'
            }


# ============================================================================
# FastAPI Application with Lifespan
# ============================================================================

# Create service instance
service = FaceMatchService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    service.initialize()
    yield
    # Shutdown (cleanup if needed)
    pass


# Create FastAPI app with lifespan
app = FastAPI(
    title="Face Match Service",
    description="Cross-modal face detection and matching service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=StatusResponse)
async def get_status():
    """Get service status."""
    return StatusResponse(
        status="running",
        device=service.device,
        models_loaded=service.models_loaded,
        gpu_available=torch.cuda.is_available()
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/detect", response_model=DetectResponse)
async def detect_faces(request: DetectRequest):
    """Detect and align faces in image."""
    try:
        image = service.decode_image(request.image)
        result = service.detect_faces(image, request.align_size)
        return DetectResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract", response_model=ExtractResponse)
async def extract_features(request: ExtractRequest):
    """Extract features from image."""
    try:
        image = service.decode_image(request.image)
        result = service.extract_features(image, request.modality)
        return ExtractResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match", response_model=MatchResponse)
async def match_images(request: MatchRequest):
    """Match query image against gallery."""
    try:
        query_image = service.decode_image(request.query_image)
        gallery_images = [service.decode_image(img) for img in request.gallery_images]
        
        result = service.match(
            query_image,
            gallery_images,
            request.gallery_ids,
            request.top_k or 10
        )
        return MatchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """Compute similarity between two images."""
    try:
        image1 = service.decode_image(request.image1)
        image2 = service.decode_image(request.image2)
        result = service.compute_similarity(image1, image2)
        return SimilarityResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/upload")
async def detect_upload(file: UploadFile = File(...)):
    """Detect faces from uploaded file."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        result = service.detect_faces(image)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment
    port = int(os.environ.get("PORT", 8000))
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
