/**
 * 系统状态 API
 * GET /api/status
 */

import { NextResponse } from 'next/server';
import { modelManager } from '@/lib/models';

/**
 * 获取系统状态
 */
export async function GET() {
  try {
    const modelState = modelManager.getModelState();
    
    // 获取系统运行时间
    const uptime = process.uptime();
    
    // 获取内存使用情况
    const memoryUsage = process.memoryUsage();
    const memoryInfo = {
      total: memoryUsage.heapTotal,
      used: memoryUsage.heapUsed,
      free: memoryUsage.heapTotal - memoryUsage.heapUsed,
      usagePercent: (memoryUsage.heapUsed / memoryUsage.heapTotal) * 100,
    };

    return NextResponse.json({
      success: true,
      data: {
        healthy: modelState.faceDetector.loaded && 
                 modelState.featureExtractor.loaded && 
                 modelState.crossModalMatcher.loaded,
        models: modelState,
        memory: memoryInfo,
        uptime,
        timestamp: new Date(),
      },
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, message: `Failed to get status: ${error}` },
      { status: 500 }
    );
  }
}
