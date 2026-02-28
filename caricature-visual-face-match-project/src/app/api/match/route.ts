/**
 * 跨模态匹配 API
 * POST /api/match
 */

import { NextRequest, NextResponse } from 'next/server';
import { readFile, readdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { ImageModality, FeatureVector, ImageInfo } from '@/types';
import { faceMatchPipeline, featureDatabase } from '@/lib/face-match';
import { featureExtractor } from '@/lib/models';
import ModelConfig from '@/config/model.config';

/**
 * 从目录加载图像并提取特征
 */
async function loadFeaturesFromDirectory(
  dirPath: string,
  modality: ImageModality
): Promise<{ features: FeatureVector[]; images: ImageInfo[] }> {
  const features: FeatureVector[] = [];
  const images: ImageInfo[] = [];

  if (!existsSync(dirPath)) {
    return { features, images };
  }

  const files = await readdir(dirPath);
  const supportedExts = ModelConfig.dataProcessing.supportedFormats;

  for (const file of files) {
    const ext = file.split('.').pop()?.toLowerCase();
    if (!ext || !supportedExts.includes(ext)) continue;

    const filePath = path.join(dirPath, file);
    const imageBuffer = await readFile(filePath);
    const base64 = imageBuffer.toString('base64');

    // 提取特征
    const result = await featureExtractor.extract(base64, modality);
    
    if (result.success && result.features) {
      const imageInfo: ImageInfo = {
        id: result.features.imageId,
        filename: file,
        path: filePath,
        modality,
        width: 0,
        height: 0,
        format: ext,
        size: imageBuffer.length,
        uploadTime: new Date(),
      };

      features.push(result.features);
      images.push(imageInfo);
    }
  }

  return { features, images };
}

/**
 * 执行匹配
 * POST /api/match
 * 
 * Body: {
 *   queryImage: string (Base64 or file path),
 *   queryModality: 'face' | 'caricature',
 *   targetModality: 'face' | 'caricature',
 *   topK?: number,
 *   threshold?: number
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { 
      queryImage, 
      queryModality, 
      targetModality, 
      topK = ModelConfig.crossModalMatcher.topK,
      threshold = ModelConfig.crossModalMatcher.matchThreshold,
    } = body;

    // 参数验证
    if (!queryImage) {
      return NextResponse.json(
        { success: false, message: 'Query image is required' },
        { status: 400 }
      );
    }

    if (!queryModality || !['face', 'caricature'].includes(queryModality)) {
      return NextResponse.json(
        { success: false, message: 'Invalid query modality' },
        { status: 400 }
      );
    }

    if (!targetModality || !['face', 'caricature'].includes(targetModality)) {
      return NextResponse.json(
        { success: false, message: 'Invalid target modality' },
        { status: 400 }
      );
    }

    // 加载目标特征库
    const targetDir = path.join(
      process.cwd(), 
      'data', 
      targetModality === 'face' ? 'faces' : 'caricatures'
    );
    
    const { features: targetFeatures, images: targetImages } = await loadFeaturesFromDirectory(
      targetDir,
      targetModality
    );

    if (targetFeatures.length === 0) {
      return NextResponse.json(
        { success: false, message: 'No target images found in database' },
        { status: 400 }
      );
    }

    // 准备查询图像
    let queryImageData: string;
    if (queryImage.startsWith('data:')) {
      // Base64格式
      queryImageData = queryImage.split(',')[1] || queryImage;
    } else if (queryImage.startsWith('/')) {
      // 文件路径
      const fileBuffer = await readFile(queryImage);
      queryImageData = fileBuffer.toString('base64');
    } else {
      // 假设已经是Base64
      queryImageData = queryImage;
    }

    // 执行匹配
    const matchResult = await faceMatchPipeline.match(
      queryImageData,
      queryModality,
      targetFeatures,
      topK
    );

    // 补充图像路径信息
    const imageMap = new Map(targetImages.map((img) => [img.id, img]));
    matchResult.matches = matchResult.matches.map((m) => ({
      ...m,
      imagePath: imageMap.get(m.imageId)?.path || '',
      isMatch: m.similarity >= threshold,
    }));

    return NextResponse.json({
      success: matchResult.success,
      data: matchResult,
      message: matchResult.message,
    });
  } catch (error) {
    console.error('Match error:', error);
    return NextResponse.json(
      { success: false, message: `Match failed: ${error}` },
      { status: 500 }
    );
  }
}

/**
 * 计算两张图像的相似度
 * PUT /api/match
 * 
 * Body: {
 *   image1: string,
 *   modality1: 'face' | 'caricature',
 *   image2: string,
 *   modality2: 'face' | 'caricature'
 * }
 */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const { image1, modality1, image2, modality2 } = body;

    if (!image1 || !image2 || !modality1 || !modality2) {
      return NextResponse.json(
        { success: false, message: 'Missing required parameters' },
        { status: 400 }
      );
    }

    // 准备图像数据
    const prepareImage = async (image: string): Promise<string> => {
      if (image.startsWith('data:')) {
        return image.split(',')[1] || image;
      } else if (image.startsWith('/')) {
        const buffer = await readFile(image);
        return buffer.toString('base64');
      }
      return image;
    };

    const img1Data = await prepareImage(image1);
    const img2Data = await prepareImage(image2);

    const result = await faceMatchPipeline.computeImageSimilarity(
      img1Data,
      modality1,
      img2Data,
      modality2
    );

    return NextResponse.json({
      success: result.success,
      data: result.similarity ? { similarity: result.similarity } : undefined,
      message: result.message,
    });
  } catch (error) {
    console.error('Similarity computation error:', error);
    return NextResponse.json(
      { success: false, message: `Similarity computation failed: ${error}` },
      { status: 500 }
    );
  }
}
