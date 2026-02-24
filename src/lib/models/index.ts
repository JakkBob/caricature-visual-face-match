/**
 * 模型管理器
 * 
 * 统一管理所有模型的加载、状态监控和资源释放
 */

import { FaceDetector, faceDetector } from './face-detector';
import { FeatureExtractor, featureExtractor } from './feature-extractor';
import { CrossModalMatcher, crossModalMatcher } from './cross-modal-matcher';
import { ModelState, ModelStatus } from '@/types';
import ModelConfig from '../../config/model.config';

/**
 * 模型管理器类
 */
export class ModelManager {
  private static instance: ModelManager;
  private initialized: boolean = false;

  private constructor() {}

  /**
   * 获取单例实例
   */
  static getInstance(): ModelManager {
    if (!ModelManager.instance) {
      ModelManager.instance = new ModelManager();
    }
    return ModelManager.instance;
  }

  /**
   * 初始化所有模型
   * TODO: 在系统启动时调用此方法加载所有模型
   */
  async initialize(): Promise<{ success: boolean; errors: string[] }> {
    const errors: string[] = [];
    
    console.log('[ModelManager] Starting model initialization...');

    // 加载人脸检测模型
    try {
      const detectorLoaded = await faceDetector.loadModel(
        ModelConfig.faceDetector.modelPath
      );
      if (!detectorLoaded) {
        errors.push('Failed to load face detector model');
      }
    } catch (error) {
      errors.push(`Face detector error: ${error}`);
    }

    // 加载特征提取模型
    try {
      const extractorLoaded = await featureExtractor.loadModel(
        ModelConfig.featureExtractor.modelPath
      );
      if (!extractorLoaded) {
        errors.push('Failed to load feature extractor model');
      }
    } catch (error) {
      errors.push(`Feature extractor error: ${error}`);
    }

    // 加载跨模态匹配模型
    try {
      const matcherLoaded = await crossModalMatcher.loadModel(
        ModelConfig.crossModalMatcher.modelPath
      );
      if (!matcherLoaded) {
        errors.push('Failed to load cross-modal matcher model');
      }
    } catch (error) {
      errors.push(`Cross-modal matcher error: ${error}`);
    }

    this.initialized = errors.length === 0;
    
    console.log(`[ModelManager] Initialization complete. Success: ${this.initialized}`);
    if (errors.length > 0) {
      console.warn('[ModelManager] Errors:', errors);
    }

    return {
      success: this.initialized,
      errors,
    };
  }

  /**
   * 获取所有模型状态
   */
  getModelState(): ModelState {
    return {
      faceDetector: this.getDetectorStatus(),
      faceAligner: {
        loaded: ModelConfig.faceAligner.enabled,
        loading: false,
      },
      featureExtractor: this.getExtractorStatus(),
      crossModalMatcher: this.getMatcherStatus(),
    };
  }

  /**
   * 获取人脸检测器状态
   */
  getDetectorStatus(): ModelStatus {
    const status = faceDetector.getStatus();
    return {
      loaded: status.loaded,
      loading: false,
      device: status.device,
    };
  }

  /**
   * 获取特征提取器状态
   */
  getExtractorStatus(): ModelStatus {
    const status = featureExtractor.getStatus();
    return {
      loaded: status.loaded,
      loading: false,
      device: status.device,
    };
  }

  /**
   * 获取匹配器状态
   */
  getMatcherStatus(): ModelStatus {
    const status = crossModalMatcher.getStatus();
    return {
      loaded: status.loaded,
      loading: false,
      device: status.device,
    };
  }

  /**
   * 检查系统是否就绪
   */
  isReady(): boolean {
    return this.initialized;
  }

  /**
   * 释放所有模型资源
   */
  async dispose(): Promise<void> {
    console.log('[ModelManager] Disposing all models...');
    
    await faceDetector.dispose();
    await featureExtractor.dispose();
    await crossModalMatcher.dispose();
    
    this.initialized = false;
    console.log('[ModelManager] All models disposed');
  }

  /**
   * 重新加载指定模型
   */
  async reloadModel(modelName: 'detector' | 'extractor' | 'matcher'): Promise<boolean> {
    console.log(`[ModelManager] Reloading model: ${modelName}`);
    
    switch (modelName) {
      case 'detector':
        await faceDetector.dispose();
        return faceDetector.loadModel(ModelConfig.faceDetector.modelPath);
      case 'extractor':
        await featureExtractor.dispose();
        return featureExtractor.loadModel(ModelConfig.featureExtractor.modelPath);
      case 'matcher':
        await crossModalMatcher.dispose();
        return crossModalMatcher.loadModel(ModelConfig.crossModalMatcher.modelPath);
      default:
        return false;
    }
  }
}

// 导出单例实例
export const modelManager = ModelManager.getInstance();

// 导出模型实例供外部使用
export { faceDetector, featureExtractor, crossModalMatcher };
