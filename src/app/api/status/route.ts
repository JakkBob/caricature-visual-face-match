/**
 * 系统状态 API
 * GET /api/status
 * 
 * 从 Python 服务获取真实的模型状态
 */

import { NextResponse } from 'next/server';
import ModelConfig from '../../../../config/model.config';

// Python service URL
const PYTHON_SERVICE_URL = ModelConfig.pythonService.url;

/**
 * 获取系统状态
 */
export async function GET() {
  try {
    // 调用 Python 服务获取真实的模型状态
    const pythonStatusResponse = await fetch(`${PYTHON_SERVICE_URL}/`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    let pythonStatus = {
      status: 'unknown',
      device: 'unknown',
      models_loaded: {
        face_detector: false,
        cross_modal_matcher: false,
      },
      gpu_available: false,
    };

    if (pythonStatusResponse.ok) {
      pythonStatus = await pythonStatusResponse.json();
    }

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

    // 转换为前端期望的格式
    const modelState = {
      faceDetector: {
        loaded: pythonStatus.models_loaded?.face_detector ?? false,
        loading: false,
        device: pythonStatus.device,
      },
      faceAligner: {
        loaded: true, // 人脸对齐是检测器的一部分
        loading: false,
      },
      featureExtractor: {
        loaded: pythonStatus.models_loaded?.cross_modal_matcher ?? false,
        loading: false,
        device: pythonStatus.device,
      },
      crossModalMatcher: {
        loaded: pythonStatus.models_loaded?.cross_modal_matcher ?? false,
        loading: false,
        device: pythonStatus.device,
      },
    };

    // 判断系统是否健康
    const healthy = pythonStatus.status === 'running' && 
                    (pythonStatus.models_loaded?.face_detector ?? false) &&
                    (pythonStatus.models_loaded?.cross_modal_matcher ?? false);

    return NextResponse.json({
      success: true,
      data: {
        healthy,
        models: modelState,
        memory: memoryInfo,
        uptime,
        timestamp: new Date(),
        // 额外的 Python 服务信息
        pythonService: {
          status: pythonStatus.status,
          device: pythonStatus.device,
          gpuAvailable: pythonStatus.gpu_available,
        },
      },
    });
  } catch (error) {
    console.error('Failed to get status:', error);
    
    // 返回错误状态
    return NextResponse.json({
      success: false,
      data: {
        healthy: false,
        models: {
          faceDetector: { loaded: false, loading: false, error: 'Python service unavailable' },
          faceAligner: { loaded: false, loading: false },
          featureExtractor: { loaded: false, loading: false, error: 'Python service unavailable' },
          crossModalMatcher: { loaded: false, loading: false, error: 'Python service unavailable' },
        },
        memory: {
          total: 0,
          used: 0,
          free: 0,
          usagePercent: 0,
        },
        uptime: process.uptime(),
        timestamp: new Date(),
        pythonService: {
          status: 'error',
          error: String(error),
        },
      },
      message: `Failed to connect to Python service: ${error}`,
    });
  }
}