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
  MatchResult,
  ImageModality,
  FeatureVector,
} from '@/types';
import { faceMatchPipeline } from '@/lib/face-match';
import { featureExtractor } from '@/lib/models';
import ModelConfig from '@/config/model.config';

interface GroundTruthPair {
  faceId: string;
  caricatureId: string;
  isMatch: boolean;
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
    if (!existsSync(facesDir) || !existsSync(caricaturesDir)) {
      return NextResponse.json(
        { success: false, message: 'Dataset directories not found' },
        { status: 400 }
      );
    }

    // 加载人脸特征
    const faceFiles = await readdir(facesDir);
    const faceFeatures: { id: string; feature: FeatureVector; path: string }[] = [];
    
    for (const file of faceFiles) {
      const ext = file.split('.').pop()?.toLowerCase();
      if (!ext || !ModelConfig.dataProcessing.supportedFormats.includes(ext)) continue;
      
      const filePath = path.join(facesDir, file);
      const buffer = await readFile(filePath);
      const base64 = buffer.toString('base64');
      
      const result = await featureExtractor.extract(base64, 'face');
      if (result.success && result.features) {
        faceFeatures.push({
          id: file.replace(/\.[^/.]+$/, ''),
          feature: result.features,
          path: filePath,
        });
      }
    }

    // 加载漫画特征
    const caricatureFiles = await readdir(caricaturesDir);
    const caricatureFeatures: { id: string; feature: FeatureVector; path: string }[] = [];
    
    for (const file of caricatureFiles) {
      const ext = file.split('.').pop()?.toLowerCase();
      if (!ext || !ModelConfig.dataProcessing.supportedFormats.includes(ext)) continue;
      
      const filePath = path.join(caricaturesDir, file);
      const buffer = await readFile(filePath);
      const base64 = buffer.toString('base64');
      
      const result = await featureExtractor.extract(base64, 'caricature');
      if (result.success && result.features) {
        caricatureFeatures.push({
          id: file.replace(/\.[^/.]+$/, ''),
          feature: result.features,
          path: filePath,
        });
      }
    }

    if (faceFeatures.length === 0 || caricatureFeatures.length === 0) {
      return NextResponse.json(
        { success: false, message: 'No valid images found in dataset' },
        { status: 400 }
      );
    }

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

    // 执行匹配评估
    const evaluationResults: { 
      queryId: string; 
      matches: { targetId: string; similarity: number; rank: number }[]; 
      groundTruth: string;
    }[] = [];

    // 漫画 -> 人脸 匹配
    for (const caricature of caricatureFeatures) {
      const gt = groundTruth.find((g) => g.caricatureId === caricature.id);
      if (!gt) continue;

      const matchResult = await faceMatchPipeline.match(
        Buffer.from(JSON.stringify(caricature.feature.vector)).toString('base64'),
        'caricature',
        faceFeatures.map((f) => f.feature),
        10
      );

      if (matchResult.success) {
        evaluationResults.push({
          queryId: caricature.id,
          matches: matchResult.matches.map((m, i) => ({
            targetId: faceFeatures.find((f) => f.feature.imageId === m.imageId)?.id || m.imageId,
            similarity: m.similarity,
            rank: i + 1,
          })),
          groundTruth: gt.faceId,
        });
      }
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
