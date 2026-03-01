# Face Match Python Service

Python backend service for cross-modal face detection and matching.

## Features

- **Face Detection**: YOLOv5-Face based face detection with 5 facial landmarks
- **Face Alignment**: ArcFace-style face alignment to 224x224
- **Feature Extraction**: Combined FaceNet + CLIP feature extraction
- **Cross-Modal Matching**: Match caricatures to real face photos

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU inference)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Model Files

Place the following model files in the `models/` directory:

1. `yolov5l6_best.pt` - YOLOv5-Face model (required)
2. `cross_modal_matcher.pt` - Trained cross-modal matcher weights (optional)

## Usage

### Start the Service

```bash
# Using the startup script
./start.sh

# Or directly with Python
python main.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FACE_DETECTOR_MODEL` | `./models/yolov5l6_best.pt` | Path to face detector model |
| `CROSS_MODAL_MODEL` | `./models/cross_modal_matcher.pt` | Path to cross-modal model |
| `DEVICE` | `cuda` | Device to use (cuda/cpu) |
| `PORT` | `8000` | Service port |

## API Endpoints

### GET /
Get service status.

### GET /health
Health check endpoint.

### POST /detect
Detect and align faces in an image.

**Request:**
```json
{
  "image": "base64_encoded_image",
  "align_size": 224
}
```

**Response:**
```json
{
  "success": true,
  "faces": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "landmarks": [[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x5,y5]]
    }
  ],
  "aligned_image": "base64_encoded_aligned_face"
}
```

### POST /extract
Extract features from an image.

**Request:**
```json
{
  "image": "base64_encoded_image",
  "modality": "face"
}
```

**Response:**
```json
{
  "success": true,
  "feature": [0.1, 0.2, ...],
  "dimension": 512
}
```

### POST /match
Match a query image against a gallery.

**Request:**
```json
{
  "query_image": "base64_encoded_image",
  "gallery_images": ["base64_1", "base64_2", ...],
  "gallery_ids": ["id1", "id2", ...],
  "top_k": 10
}
```

**Response:**
```json
{
  "success": true,
  "matches": [
    {
      "index": 0,
      "id": "image1.jpg",
      "similarity": 0.85,
      "rank": 1,
      "is_match": true
    }
  ],
  "query_feature": [0.1, 0.2, ...]
}
```

### POST /similarity
Compute similarity between two images.

**Request:**
```json
{
  "image1": "base64_encoded_image",
  "image2": "base64_encoded_image"
}
```

**Response:**
```json
{
  "success": true,
  "similarity": 0.85
}
```

## Architecture

```
python_service/
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── start.sh                   # Startup script
├── models/                    # Model files directory
├── face_detection/            # Face detection module
│   ├── __init__.py
│   └── face_detector.py       # YOLOv5-Face implementation
└── cross_modal/               # Cross-modal matching module
    ├── __init__.py
    └── matcher.py             # Cross-modal matcher implementation
```

## Integration with Next.js

The Python service is designed to work with the Next.js frontend. Set the `PYTHON_SERVICE_URL` environment variable in your Next.js application:

```bash
export PYTHON_SERVICE_URL=http://localhost:8000
```
