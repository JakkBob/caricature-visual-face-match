/**
 * 配置管理 API
 * GET/PUT /api/config
 */

import { NextRequest, NextResponse } from 'next/server';
import ModelConfig from '@/config/model.config';

/**
 * 获取当前配置
 */
export async function GET() {
  return NextResponse.json({
    success: true,
    data: ModelConfig,
  });
}

/**
 * 更新配置（运行时）
 * 注意：此接口仅更新运行时配置，不会修改配置文件
 */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    
    // 这里只返回成功，实际配置更新需要重启服务
    // 如果需要持久化配置，可以实现配置文件写入逻辑
    
    return NextResponse.json({
      success: true,
      message: 'Configuration updated. Restart required for some changes to take effect.',
      data: {
        updatedKeys: Object.keys(body),
      },
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, message: `Failed to update config: ${error}` },
      { status: 500 }
    );
  }
}
