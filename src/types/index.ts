/**
 * 跨模态漫画-视觉人脸识别系统 - 类型定义
 */

// ==================== 图像相关类型 ====================

/** 图像模态类型 */
export type ImageModality = 'face' | 'caricature';

/** 图像信息 */
export interface ImageInfo {
  id: string;
  filename: string;
  path: string;
  modality: ImageModality;
  width: number;
  height: number;
  format: string;
  size: number;
  uploadTime: Date;
}

/** 人脸检测结果 */
export interface FaceDetectionResult {
  success: boolean;
  faces: DetectedFace[];
  message?: string;
}

/** 检测到的人脸 */
export interface DetectedFace {
  id: string;
  bbox: BoundingBox;
  confidence: number;
  landmarks?: Landmark[];
  alignedImage?: string; // Base64编码的对齐后图像
}

/** 边界框 */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** 关键点 */
export interface Landmark {
  x: number;
  y: number;
  label?: string;
}

// ==================== 特征相关类型 ====================

/** 特征提取结果 */
export interface FeatureExtractionResult {
  success: boolean;
  features?: FeatureVector;
  message?: string;
}

/** 特征向量 */
export interface FeatureVector {
  id: string;
  vector: number[];
  dimension: number;
  modality: ImageModality;
  imageId: string;
  extractTime: Date;
}

// ==================== 匹配相关类型 ====================

/** 匹配请求 */
export interface MatchRequest {
  queryImageId: string;
  queryModality: ImageModality;
  targetModality: ImageModality;
  topK?: number;
  threshold?: number;
}

/** 匹配结果 */
export interface MatchResult {
  success: boolean;
  queryImageId: string;
  queryModality: ImageModality;
  matches: MatchedPair[];
  message?: string;
  processTime: number;
}

/** 匹配对 */
export interface MatchedPair {
  imageId: string;
  imagePath: string;
  similarity: number;
  rank: number;
  isMatch: boolean; // 是否超过阈值
}

/** 批量匹配结果 */
export interface BatchMatchResult {
  success: boolean;
  results: MatchResult[];
  summary: {
    total: number;
    correct: number;
    accuracy: number;
  };
}

// ==================== 评估相关类型 ====================

/** 评估指标 */
export interface EvaluationMetrics {
  // Rank-K 准确率
  rank1: number;
  rank5: number;
  rank10: number;
  
  // 平均精度
  mAP: number;
  
  // ROC相关
  auc: number;
  
  // 混淆矩阵相关
  truePositive: number;
  falsePositive: number;
  trueNegative: number;
  falseNegative: number;
  precision: number;
  recall: number;
  f1Score: number;
}

/** 评估请求 */
export interface EvaluationRequest {
  datasetPath?: string;
  groundTruthPath?: string;
  saveResults?: boolean;
}

/** 评估结果 */
export interface EvaluationResult {
  success: boolean;
  metrics: EvaluationMetrics;
  confusionMatrix?: number[][];
  rocCurve?: ROCCurveData;
  reportPath?: string;
  message?: string;
}

/** ROC曲线数据 */
export interface ROCCurveData {
  fpr: number[]; // False Positive Rate
  tpr: number[]; // True Positive Rate
  thresholds: number[];
}

// ==================== 数据集相关类型 ====================

/** 数据集信息 */
export interface DatasetInfo {
  id: string;
  name: string;
  description?: string;
  faceCount: number;
  caricatureCount: number;
  pairCount: number;
  createTime: Date;
}

/** 图像对 */
export interface ImagePair {
  id: string;
  faceImageId: string;
  caricatureImageId: string;
  isMatch: boolean; // 是否为同一身份
  label?: string;
}

// ==================== 系统状态类型 ====================

/** 模型状态 */
export interface ModelState {
  faceDetector: ModelStatus;
  faceAligner: ModelStatus;
  featureExtractor: ModelStatus;
  crossModalMatcher: ModelStatus;
}

/** 模型状态详情 */
export interface ModelStatus {
  loaded: boolean;
  loading: boolean;
  error?: string;
  device?: string;
  memoryUsage?: number;
}

/** 系统状态 */
export interface SystemStatus {
  healthy: boolean;
  models: ModelState;
  gpu: GPUInfo[];
  memory: MemoryInfo;
  uptime: number;
}

/** GPU信息 */
export interface GPUInfo {
  id: number;
  name: string;
  memoryTotal: number;
  memoryUsed: number;
  memoryFree: number;
  utilization: number;
}

/** 内存信息 */
export interface MemoryInfo {
  total: number;
  used: number;
  free: number;
  usagePercent: number;
}

// ==================== API响应类型 ====================

/** 通用API响应 */
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

/** 分页响应 */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}
