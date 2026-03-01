/**
 * 跨模态漫画-视觉人脸识别系统 - 模型配置文件
 * 
 * 说明：
 * 1. 请根据您的模型实际情况修改以下配置参数
 * 2. 模型文件请放置在 /models 目录下
 * 3. 配置修改后需重启服务生效
 * 4. Python后端服务默认运行在 http://localhost:8000
 */

export const ModelConfig = {
  // ==================== Python后端服务配置 ====================
  pythonService: {
    // Python服务URL
    url: process.env.PYTHON_SERVICE_URL || 'http://localhost:8000',
    
    // 请求超时时间（毫秒）
    timeout: 30000,
  },

  // ==================== 人脸检测模型配置 ====================
  faceDetector: {
    // 模型文件路径（相对于python_service目录）
    modelPath: './models/yolov5l6_best.pt',
    
    // 模型类型
    modelType: 'pytorch' as const,
    
    // 输入图像尺寸
    inputSize: {
      width: 640,
      height: 640,
    },
    
    // 人脸对齐输出尺寸
    alignSize: 224,
    
    // 检测置信度阈值
    confidenceThreshold: 0.2,
    
    // 非极大值抑制阈值
    nmsThreshold: 0.5,
    
    // 是否使用GPU
    useGPU: true,
    
    // GPU设备ID（多GPU情况下）
    deviceID: 0,
  },

  // ==================== 人脸对齐模型配置 ====================
  faceAligner: {
    // 是否启用人脸对齐
    enabled: true,
    
    // 对齐后输出尺寸
    outputSize: {
      width: 224,
      height: 224,
    },
    
    // 关键点数量（通常为5）
    numLandmarks: 5,
    
    // 对齐方法：'arcface' | 'similarity'
    alignMethod: 'arcface' as const,
  },

  // ==================== 特征提取模型配置 ====================
  featureExtractor: {
    // 跨模态匹配模型路径（相对于python_service目录）
    modelPath: './models/cross_modal_matcher.pt',
    
    // 模型类型
    modelType: 'pytorch' as const,
    
    // FaceNet输入尺寸
    facenetInputSize: {
      width: 160,
      height: 160,
    },
    
    // CLIP输入尺寸
    clipInputSize: {
      width: 224,
      height: 224,
    },
    
    // 输出特征维度
    featureDim: 512,
    
    // 是否归一化特征向量
    normalizeFeatures: true,
    
    // 是否使用GPU
    useGPU: true,
    
    // GPU设备ID
    deviceID: 0,
    
    // 批处理大小
    batchSize: 32,
  },

  // ==================== 跨模态匹配模型配置 ====================
  crossModalMatcher: {
    // 模型文件路径
    modelPath: './models/cross_modal_matcher.pt',
    
    // 模型类型
    modelType: 'pytorch' as const,
    
    // 是否使用GPU
    useGPU: true,
    
    // GPU设备ID
    deviceID: 0,
    
    // 相似度计算方法：'cosine' | 'euclidean' | 'custom'
    similarityMethod: 'cosine' as const,
    
    // 匹配阈值（高于此值认为是同一身份）
    matchThreshold: 0.5,
    
    // Top-K返回结果数量
    topK: 10,
    
    // 嵌入维度
    embeddingDim: 512,
  },

  // ==================== 数据处理配置 ====================
  dataProcessing: {
    // 支持的图像格式
    supportedFormats: ['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    
    // 最大图像尺寸（超过会自动缩放）
    maxImageSize: 2048,
    
    // FaceNet图像归一化参数
    facenetNormalization: {
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
    },
    
    // CLIP图像归一化参数
    clipNormalization: {
      mean: [0.48145466, 0.4578275, 0.40821073],
      std: [0.26862954, 0.26130258, 0.27577711],
    },
    
    // 数据增强（训练时使用，推理时可选）
    augmentation: {
      enabled: false,
      horizontalFlip: false,
      rotation: 0,
      brightness: 0,
      contrast: 0,
    },
  },

  // ==================== 系统配置 ====================
  system: {
    // 最大上传文件大小（MB）
    maxUploadSize: 10,
    
    // 临时文件目录
    tempDir: './temp',
    
    // 结果缓存时间（秒）
    cacheTimeout: 3600,
    
    // 日志级别：'debug' | 'info' | 'warn' | 'error'
    logLevel: 'info' as const,
    
    // 并发处理数量
    maxConcurrency: 4,
  },
} as const;

// 导出类型定义
export type PythonServiceConfig = typeof ModelConfig.pythonService;
export type FaceDetectorConfig = typeof ModelConfig.faceDetector;
export type FaceAlignerConfig = typeof ModelConfig.faceAligner;
export type FeatureExtractorConfig = typeof ModelConfig.featureExtractor;
export type CrossModalMatcherConfig = typeof ModelConfig.crossModalMatcher;
export type DataProcessingConfig = typeof ModelConfig.dataProcessing;
export type SystemConfig = typeof ModelConfig.system;

export default ModelConfig;
