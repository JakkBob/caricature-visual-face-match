#!/bin/bash

# Face Match Python Service Startup Script

# Set environment variables
export FACE_DETECTOR_MODEL=${FACE_DETECTOR_MODEL:-"./models/yolov5l6_best.pt"}
export CROSS_MODAL_MODEL=${CROSS_MODAL_MODEL:-"./models/cross_modal_matcher.pt"}
export DEVICE=${DEVICE:-"cuda"}
export PORT=${PORT:-8000}

# Check if models exist
if [ ! -f "$FACE_DETECTOR_MODEL" ]; then
    echo "Warning: Face detector model not found at $FACE_DETECTOR_MODEL"
    echo "Please place the model file in the models directory"
fi

# Start the service
echo "Starting Face Match Service on port $PORT..."
echo "Device: $DEVICE"
echo "Face Detector Model: $FACE_DETECTOR_MODEL"
echo "Cross-Modal Model: $CROSS_MODAL_MODEL"

cd "$(dirname "$0")"
python main.py
