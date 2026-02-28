/**
 * 实验评估 API
 * POST /api/evaluate
 */

import { NextRequest, NextResponse } from 'next/server';
import { readFile, readdir, writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { 
  EvaluationMetrics, 
  EvaluationResult,
  ImageModality,
} from '@/types';
import ModelConfig from '../../config/model.config';

// Python service URL
const PYTHON_SERVICE_URL = ModelConfig.pythonService.url;

interface GroundTruthPair {
  faceId: string;
  caricatureId: string;
  isMatch: boolean;
}

/**
 * 调用Python服务提取特征
 */
async function extractFeaturesFromPythonService(
  base64Image: string,
  modality: ImageModality
): Promise<{ success: boolean; feature?: number[]; message?: string }> {
  try {
    const response = await fetch(`${PYTHON_SERVICE_URL}/extract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64Image,
        modality: modality,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      return { success: false, message: error };
    }

    const result = await response.json();
    return {
      success: result.success,
      feature: result.feature,
      message: result.message,
    };
  } catch (error) {
    return { success: false, message: String(error) };
  }
}

/**
 * 调用Python服务计算相似度
 */
async function computeSimilarityFromPythonService(
  image1Base64: string,
  image2Base64: string
): Promise<{ success: boolean; similarity?: number; message?: string }> {
  try {
    const response = await fetch(`${PYTHON_SERVICE_URL}/similarity`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image1: image1Base64,
        image2: image2Base64,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      return { success: false, message: error };
    }

    const result = await response.json();
    return {
      success: result.success,
      similarity: result.similarity,
      message: result.message,
    };
  } catch (error) {
    return { success: false, message: String(error) };
  }
}

/**
 * 计算余弦相似度
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * 计算评估指标
 */
function calculateMetrics(
  results: { queryId: string; matches: { targetId: string; similarity: number; rank: number }[]; groundTruth: string }[],
  threshold: number
): EvaluationMetrics {
  let truePositive = 0;
  let falsePositive = 0;
  let trueNegative = 0;
  let falseNegative = 0;
  
  const rankHits: { [key: number]: number } = { 1: 0, 5: 0, 10: 0 };
  let totalQueries = results.length;
  let apSum = 0;

  for (const result of results) {
    const groundTruth = result.groundTruth;
    const matches = result.matches;

    // Rank-K 准确率
    for (const k of [1, 5, 10]) {
      const topKMatches = matches.slice(0, k);
      if (topKMatches.some((m) => m.targetId === groundTruth)) {
        rankHits[k]++;
      }
    }

    // 计算 AP (Average Precision)
    let relevantCount = 0;
    let precisionSum = 0;
    for (let i = 0; i < matches.length; i++) {
      if (matches[i].targetId === groundTruth) {
        relevantCount++;
        precisionSum += relevantCount / (i + 1);
      }
    }
    if (relevantCount > 0) {
      apSum += precisionSum / relevantCount;
    }

    // 混淆矩阵计算（基于阈值）
    const topMatch = matches[0];
    if (topMatch) {
      const isPredictedMatch = topMatch.similarity >= threshold;
      const isActualMatch = topMatch.targetId === groundTruth;

      if (isPredictedMatch && isActualMatch) truePositive++;
      else if (isPredictedMatch && !isActualMatch) falsePositive++;
      else if (!isPredictedMatch && isActualMatch) falseNegative++;
      else trueNegative++;
    }
  }

  const precision = truePositive / (truePositive + falsePositive) || 0;
  const recall = truePositive / (truePositive + falseNegative) || 0;
  const f1Score = 2 * (precision * recall) / (precision + recall) || 0;

  return {
    rank1: rankHits[1] / totalQueries,
    rank5: rankHits[5] / totalQueries,
    rank10: rankHits[10] / totalQueries,
    mAP: apSum / totalQueries,
    auc: 0, // TODO: 实现AUC计算
    truePositive,
    falsePositive,
    trueNegative,
    falseNegative,
    precision,
    recall,
    f1Score,
  };
}

/**
 * 执行评估
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { 
      datasetPath,
      groundTruthPath,
      saveResults = true,
    } = body;

    // 默认数据集路径
    const facesDir = datasetPath 
      ? path.join(datasetPath, 'faces')
      : path.join(process.cwd(), 'data', 'faces');
    const caricaturesDir = datasetPath
      ? path.join(datasetPath, 'caricatures')
      : path.join(process.cwd(), 'data', 'caricatures');

    // 检查目录是否存在
    if (!existsSync(facesDir)) {
      return NextResponse.json(
        { success: false, message: `Faces directory not found: ${facesDir}. Please add face images to data/faces/ directory.` },
        { status: 400 }
      );
    }
    
    if (!existsSync(caricaturesDir)) {
      return NextResponse.json(
        { success: false, message: `Caricatures directory not found: ${caricaturesDir}. Please add caricature images to data/caricatures/ directory.` },
        { status: 400 }
      );
    }

    // 加载人脸图像
    const faceFiles = await readdir(facesDir);
    const faceImages: { id: string; base64: string; path: string }[] = [];
    
    for (const file of faceFiles) {
      const ext = file.split('.').pop()?.toLowerCase();
      if (!ext || !ModelConfig.dataProcessing.supportedFormats.includes(ext)) continue;
      
      const filePath = path.join(facesDir, file);
      const buffer = await readFile(filePath);
      const base64 = buffer.toString('base64');
      
      faceImages.push({
        id: file.replace(/\.[^/.]+$/, ''),
        base64,
        path: filePath,
      });
    }

    // 加载漫画图像
    const caricatureFiles = await readdir(caricaturesDir);
    const caricatureImages: { id: string; base64: string; path: string }[] = [];
    
    for (const file of caricatureFiles) {
      const ext = file.split('.').pop()?.toLowerCase();
      if (!ext || !ModelConfig.dataProcessing.supportedFormats.includes(ext)) continue;
      
      const filePath = path.join(caricaturesDir, file);
      const buffer = await readFile(filePath);
      const base64 = buffer.toString('base64');
      
      caricatureImages.push({
        id: file.replace(/\.[^/.]+$/, ''),
        base64,
        path: filePath,
      });
    }

    if (faceImages.length === 0) {
      return NextResponse.json(
        { success: false, message: `No valid face images found in ${facesDir}. Supported formats: ${ModelConfig.dataProcessing.supportedFormats.join(', ')}` },
        { status: 400 }
      );
    }
    
    if (caricatureImages.length === 0) {
      return NextResponse.json(
        { success: false, message: `No valid caricature images found in ${caricaturesDir}. Supported formats: ${ModelConfig.dataProcessing.supportedFormats.join(', ')}` },
        { status: 400 }
      );
    }

    console.log(`[Evaluation] Found ${faceImages.length} face images and ${caricatureImages.length} caricature images`);

    // 提取人脸特征
    console.log('[Evaluation] Extracting face features...');
    const faceFeatures: { id: string; feature: number[]; base64: string }[] = [];
    
    for (const face of faceImages) {
      const result = await extractFeaturesFromPythonService(face.base64, 'face');
      if (result.success && result.feature) {
        faceFeatures.push({
          id: face.id,
          feature: result.feature,
          base64: face.base64,
        });
      } else {
        console.warn(`[Evaluation] Failed to extract features for face ${face.id}: ${result.message}`);
      }
    }

    // 提取漫画特征
    console.log('[Evaluation] Extracting caricature features...');
    const caricatureFeatures: { id: string; feature: number[]; base64: string }[] = [];
    
    for (const caricature of caricatureImages) {
      const result = await extractFeaturesFromPythonService(caricature.base64, 'caricature');
      if (result.success && result.feature) {
        caricatureFeatures.push({
          id: caricature.id,
          feature: result.feature,
          base64: caricature.base64,
        });
      } else {
        console.warn(`[Evaluation] Failed to extract features for caricature ${caricature.id}: ${result.message}`);
      }
    }

    if (faceFeatures.length === 0 || caricatureFeatures.length === 0) {
      return NextResponse.json(
        { success: false, message: 'Failed to extract features from images. Make sure Python service is running and models are loaded.' },
        { status: 500 }
      );
    }

    console.log(`[Evaluation] Extracted features for ${faceFeatures.length} faces and ${caricatureFeatures.length} caricatures`);

    // 加载Ground Truth（如果有）
    let groundTruth: GroundTruthPair[] = [];
    if (groundTruthPath && existsSync(groundTruthPath)) {
      const gtContent = await readFile(groundTruthPath, 'utf-8');
      groundTruth = JSON.parse(gtContent);
    } else {
      // 自动生成Ground Truth（假设文件名相同的是匹配对）
      for (const face of faceFeatures) {
        const matchingCaricature = caricatureFeatures.find(
          (c) => c.id === face.id || c.id.includes(face.id) || face.id.includes(c.id)
        );
        if (matchingCaricature) {
          groundTruth.push({
            faceId: face.id,
            caricatureId: matchingCaricature.id,
            isMatch: true,
          });
        }
      }
    }

    if (groundTruth.length === 0) {
      return NextResponse.json(
        { success: false, message: 'No ground truth pairs found. Please ensure face and caricature images have matching filenames.' },
        { status: 400 }
      );
    }

    console.log(`[Evaluation] Found ${groundTruth.length} ground truth pairs`);

    // 执行匹配评估
    const evaluationResults: { 
      queryId: string; 
      matches: { targetId: string; similarity: number; rank: number }[]; 
      groundTruth: string;
    }[] = [];

    // 漫画 -> 人脸 匹配
    console.log('[Evaluation] Running matching evaluation...');
    for (const caricature of caricatureFeatures) {
      const gt = groundTruth.find((g) => g.caricatureId === caricature.id);
      if (!gt) continue;

      // 计算与所有人脸的相似度
      const similarities: { targetId: string; similarity: number }[] = [];
      
      for (const face of faceFeatures) {
        const similarity = cosineSimilarity(caricature.feature, face.feature);
        similarities.push({
          targetId: face.id,
          similarity,
        });
      }

      // 按相似度排序
      similarities.sort((a, b) => b.similarity - a.similarity);

      evaluationResults.push({
        queryId: caricature.id,
        matches: similarities.slice(0, 10).map((m, i) => ({
          targetId: m.targetId,
          similarity: m.similarity,
          rank: i + 1,
        })),
        groundTruth: gt.faceId,
      });
    }

    // 计算评估指标
    const metrics = calculateMetrics(
      evaluationResults,
      ModelConfig.crossModalMatcher.matchThreshold
    );

    // 生成混淆矩阵
    const confusionMatrix = [
      [metrics.truePositive, metrics.falsePositive],
      [metrics.falseNegative, metrics.trueNegative],
    ];

    // 保存结果
    let reportPath: string | undefined;
    if (saveResults) {
      const reportDir = path.join(process.cwd(), 'reports');
      if (!existsSync(reportDir)) {
        await mkdir(reportDir, { recursive: true });
      }

      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      reportPath = path.join(reportDir, `evaluation_${timestamp}.json`);

      const report = {
        timestamp: new Date(),
        metrics,
        confusionMatrix,
        results: evaluationResults,
        config: {
          threshold: ModelConfig.crossModalMatcher.matchThreshold,
          topK: ModelConfig.crossModalMatcher.topK,
          faceCount: faceFeatures.length,
          caricatureCount: caricatureFeatures.length,
          groundTruthCount: groundTruth.length,
        },
      };

      await writeFile(reportPath, JSON.stringify(report, null, 2));
    }

    const result: EvaluationResult = {
      success: true,
      metrics,
      confusionMatrix,
      reportPath,
    };

    return NextResponse.json({
      success: true,
      data: result,
    });
  } catch (error) {
    console.error('Evaluation error:', error);
    return NextResponse.json(
      { success: false, message: `Evaluation failed: ${error}` },
      { status: 500 }
    );
  }
}

/**
 * 获取评估历史
 */
export async function GET(request: NextRequest) {
  try {
    const reportDir = path.join(process.cwd(), 'reports');
    
    if (!existsSync(reportDir)) {
      return NextResponse.json({
        success: true,
        data: { reports: [] },
      });
    }

    const files = await readdir(reportDir);
    const reports = files
      .filter((f) => f.startsWith('evaluation_') && f.endsWith('.json'))
      .map((f) => ({
        filename: f,
        path: path.join(reportDir, f),
      }));

    return NextResponse.json({
      success: true,
      data: { reports },
    });
  } catch (error) {
    console.error('Get reports error:', error);
    return NextResponse.json(
      { success: false, message: `Failed to get reports: ${error}` },
      { status: 500 }
    );
  }
}
