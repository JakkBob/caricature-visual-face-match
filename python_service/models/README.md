# 模型文件目录

请将您的PyTorch模型文件放置在此目录下：

## 模型文件说明

### 1. 人脸检测模型 (必需)
- **文件名**: `yolov5l6_best.pt`
- **来源**: AnyFace-face-detect/yolov5-face/weights/
- **说明**: YOLOv5-Face模型，用于检测人脸和5个关键点

### 2. 跨模态匹配模型 (可选)
- **文件名**: `cross_modal_matcher.pt`
- **说明**: 训练好的跨模态人脸匹配模型权重
- **注意**: 如果没有此文件，系统会使用预训练的FaceNet和CLIP模型

## 模型文件放置位置

将模型文件放置在以下两个位置之一：

### Python服务目录（推荐）
```
python_service/models/
├── yolov5l6_best.pt           # 人脸检测模型
└── cross_modal_matcher.pt     # 跨模态匹配模型（可选）
```

### 项目根目录
```
models/
├── face_detector.pt           # 人脸检测模型
├── feature_extractor.pt       # 特征提取模型
└── cross_modal_matcher.pt     # 跨模态匹配模型（可选）
```

## 环境变量配置

可以通过环境变量指定模型路径：

```bash
# Python服务配置
export FACE_DETECTOR_MODEL=./models/yolov5l6_best.pt
export CROSS_MODAL_MODEL=./models/cross_modal_matcher.pt
export DEVICE=cuda  # 或 cpu
export PORT=8000
```

## 模型下载

### YOLOv5-Face模型
从仓库的 `AnyFace-face-detect/yolov5-face/weights/` 目录获取：
- `yolov5l6_best.pt`

### 预训练模型
以下模型会自动下载：
- FaceNet (casia-webface 或 vggface2)
- CLIP (ViT-B/32)

## 注意事项

1. 确保模型文件与代码版本匹配
2. GPU推理需要CUDA环境
3. 首次运行时会自动下载预训练模型
