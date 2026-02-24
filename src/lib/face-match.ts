/**
 * 跨模态人脸匹配核心逻辑
 * 
 * 整合人脸检测、特征提取和匹配功能
 */

import { v4 as uuidv4 } from 'uuid';
import {
  ImageInfo,
  ImageModality,
  FaceDetectionResult,
  FeatureExtractionResult,
  MatchResult,
  MatchedPair,
  FeatureVector,
} from '@/types';
import { faceDetector, featureExtractor, crossModalMatcher } from './models';
import ModelConfig from '../../config/model.config';

/**
 * 图像处理管道
 */
export class FaceMatchPipeline {
  private static instance: FaceMatchPipeline;

  private constructor() {}

  static getInstance(): FaceMatchPipeline {
    if (!FaceMatchPipeline.instance) {
      FaceMatchPipeline.instance = new FaceMatchPipeline();
    }
    return FaceMatchPipeline.instance;
  }

  /**
   * 处理单张图像：检测人脸 -> 对齐 -> 提取特征
   * @param image 图像数据（Buffer或Base64）
   * @param modality 图像模态
   * @returns 处理结果
   */
  async processImage(
    image: Buffer | string,
    modality: ImageModality
  ): Promise<{
    success: boolean;
    detection?: FaceDetectionResult;
    features?: FeatureExtractionResult;
    message?: string;
  }> {
    try {
      // Step 1: 人脸检测
      const detection = await faceDetector.detect(image);
      
      if (!detection.success || detection.faces.length === 0) {
        return {
          success: false,
          detection,
          message: 'No face detected in the image',
        };
      }

      // Step 2: 特征提取（使用检测到的第一张人脸）
      // TODO: 如果需要对齐，在此处添加对齐逻辑
      const alignedImage = detection.faces[0].alignedImage || image;
      
      const features = await featureExtractor.extract(alignedImage, modality);

      return {
        success: features.success,
        detection,
        features,
        message: features.success ? undefined : 'Feature extraction failed',
      };
    } catch (error) {
      return {
        success: false,
        message: `Processing failed: ${error}`,
      };
    }
  }

  /**
   * 执行跨模态匹配
   * @param queryImage 查询图像
   * @param queryModality 查询图像模态
   * @param targetFeatures 目标特征库
   * @param topK 返回Top-K结果
   * @returns 匹配结果
   */
  async match(
    queryImage: Buffer | string,
    queryModality: ImageModality,
    targetFeatures: FeatureVector[],
    topK?: number
  ): Promise<MatchResult> {
    const startTime = Date.now();

    try {
      // 处理查询图像
      const processResult = await this.processImage(queryImage, queryModality);
      
      if (!processResult.success || !processResult.features?.features) {
        return {
          success: false,
          queryImageId: '',
          queryModality,
          matches: [],
          message: processResult.message || 'Failed to process query image',
          processTime: Date.now() - startTime,
        };
      }

      // 执行匹配
      const matchResult = await crossModalMatcher.match(
        processResult.features.features,
        targetFeatures
      );

      return {
        ...matchResult,
        processTime: Date.now() - startTime,
      };
    } catch (error) {
      return {
        success: false,
        queryImageId: '',
        queryModality,
        matches: [],
        message: `Matching failed: ${error}`,
        processTime: Date.now() - startTime,
      };
    }
  }

  /**
   * 批量匹配
   * @param queryImages 查询图像数组
   * @param queryModality 查询图像模态
   * @param targetFeatures 目标特征库
   * @param topK 返回Top-K结果
   * @returns 匹配结果数组
   */
  async batchMatch(
    queryImages: (Buffer | string)[],
    queryModality: ImageModality,
    targetFeatures: FeatureVector[],
    topK?: number
  ): Promise<MatchResult[]> {
    const results: MatchResult[] = [];
    
    for (const image of queryImages) {
      const result = await this.match(image, queryModality, targetFeatures, topK);
      results.push(result);
    }
    
    return results;
  }

  /**
   * 计算两张图像的相似度
   * @param image1 图像1
   * @param modality1 图像1模态
   * @param image2 图像2
   * @param modality2 图像2模态
   * @returns 相似度分数
   */
  async computeImageSimilarity(
    image1: Buffer | string,
    modality1: ImageModality,
    image2: Buffer | string,
    modality2: ImageModality
  ): Promise<{ success: boolean; similarity?: number; message?: string }> {
    try {
      // 处理两张图像
      const [result1, result2] = await Promise.all([
        this.processImage(image1, modality1),
        this.processImage(image2, modality2),
      ]);

      if (!result1.success || !result1.features?.features) {
        return { success: false, message: 'Failed to process first image' };
      }

      if (!result2.success || !result2.features?.features) {
        return { success: false, message: 'Failed to process second image' };
      }

      // 计算相似度
      const similarity = await crossModalMatcher.computeSimilarity(
        result1.features.features.vector,
        result2.features.features.vector
      );

      return { success: true, similarity };
    } catch (error) {
      return { success: false, message: `Similarity computation failed: ${error}` };
    }
  }
}

/**
 * 特征库管理
 */
export class FeatureDatabase {
  private features: Map<string, FeatureVector> = new Map();
  private imageInfo: Map<string, ImageInfo> = new Map();

  /**
   * 添加特征到库
   */
  addFeature(feature: FeatureVector, imageInfo: ImageInfo): void {
    this.features.set(feature.id, feature);
    this.imageInfo.set(feature.imageId, imageInfo);
  }

  /**
   * 批量添加特征
   */
  addFeatures(items: { feature: FeatureVector; imageInfo: ImageInfo }[]): void {
    for (const item of items) {
      this.addFeature(item.feature, item.imageInfo);
    }
  }

  /**
   * 获取所有特征
   */
  getAllFeatures(): FeatureVector[] {
    return Array.from(this.features.values());
  }

  /**
   * 按模态获取特征
   */
  getFeaturesByModality(modality: ImageModality): FeatureVector[] {
    return this.getAllFeatures().filter((f) => f.modality === modality);
  }

  /**
   * 获取图像信息
   */
  getImageInfo(imageId: string): ImageInfo | undefined {
    return this.imageInfo.get(imageId);
  }

  /**
   * 删除特征
   */
  removeFeature(featureId: string): boolean {
    const feature = this.features.get(featureId);
    if (feature) {
      this.features.delete(featureId);
      this.imageInfo.delete(feature.imageId);
      return true;
    }
    return false;
  }

  /**
   * 清空特征库
   */
  clear(): void {
    this.features.clear();
    this.imageInfo.clear();
  }

  /**
   * 获取特征库统计信息
   */
  getStats(): { total: number; faceCount: number; caricatureCount: number } {
    const all = this.getAllFeatures();
    return {
      total: all.length,
      faceCount: all.filter((f) => f.modality === 'face').length,
      caricatureCount: all.filter((f) => f.modality === 'caricature').length,
    };
  }
}

// 导出单例实例
export const faceMatchPipeline = FaceMatchPipeline.getInstance();
export const featureDatabase = new FeatureDatabase();
