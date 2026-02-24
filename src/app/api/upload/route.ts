/**
 * 图像上传 API
 * POST /api/upload
 */

import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import ModelConfig from '@/config/model.config';

// 支持的图像格式
const SUPPORTED_FORMATS = ModelConfig.dataProcessing.supportedFormats;
const MAX_SIZE = ModelConfig.system.maxUploadSize * 1024 * 1024;

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const modality = formData.get('modality') as 'face' | 'caricature';

    if (!file) {
      return NextResponse.json(
        { success: false, message: 'No file uploaded' },
        { status: 400 }
      );
    }

    if (!modality || !['face', 'caricature'].includes(modality)) {
      return NextResponse.json(
        { success: false, message: 'Invalid modality. Must be "face" or "caricature"' },
        { status: 400 }
      );
    }

    // 检查文件大小
    if (file.size > MAX_SIZE) {
      return NextResponse.json(
        { success: false, message: `File size exceeds limit (${ModelConfig.system.maxUploadSize}MB)` },
        { status: 400 }
      );
    }

    // 检查文件格式
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!ext || !SUPPORTED_FORMATS.includes(ext)) {
      return NextResponse.json(
        { success: false, message: `Unsupported format. Supported: ${SUPPORTED_FORMATS.join(', ')}` },
        { status: 400 }
      );
    }

    // 生成唯一文件名
    const fileId = uuidv4();
    const filename = `${fileId}.${ext}`;
    
    // 确定存储目录
    const uploadDir = path.join(process.cwd(), 'data', modality === 'face' ? 'faces' : 'caricatures');
    
    // 确保目录存在
    if (!existsSync(uploadDir)) {
      await mkdir(uploadDir, { recursive: true });
    }

    // 保存文件
    const filePath = path.join(uploadDir, filename);
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    await writeFile(filePath, buffer);

    // 返回文件信息
    return NextResponse.json({
      success: true,
      data: {
        id: fileId,
        filename: file.name,
        storedName: filename,
        path: filePath,
        modality,
        size: file.size,
        type: file.type,
      },
    });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { success: false, message: `Upload failed: ${error}` },
      { status: 500 }
    );
  }
}

// 批量上传
export async function PUT(request: NextRequest) {
  try {
    const formData = await request.formData();
    const modality = formData.get('modality') as 'face' | 'caricature';
    const files = formData.getAll('files') as File[];

    if (!files || files.length === 0) {
      return NextResponse.json(
        { success: false, message: 'No files uploaded' },
        { status: 400 }
      );
    }

    if (!modality || !['face', 'caricature'].includes(modality)) {
      return NextResponse.json(
        { success: false, message: 'Invalid modality' },
        { status: 400 }
      );
    }

    const uploadDir = path.join(process.cwd(), 'data', modality === 'face' ? 'faces' : 'caricatures');
    
    if (!existsSync(uploadDir)) {
      await mkdir(uploadDir, { recursive: true });
    }

    const results = [];

    for (const file of files) {
      const ext = file.name.split('.').pop()?.toLowerCase();
      if (!ext || !SUPPORTED_FORMATS.includes(ext)) continue;

      const fileId = uuidv4();
      const filename = `${fileId}.${ext}`;
      const filePath = path.join(uploadDir, filename);

      const bytes = await file.arrayBuffer();
      const buffer = Buffer.from(bytes);
      await writeFile(filePath, buffer);

      results.push({
        id: fileId,
        filename: file.name,
        storedName: filename,
        path: filePath,
        modality,
        size: file.size,
      });
    }

    return NextResponse.json({
      success: true,
      data: {
        total: results.length,
        files: results,
      },
    });
  } catch (error) {
    console.error('Batch upload error:', error);
    return NextResponse.json(
      { success: false, message: `Batch upload failed: ${error}` },
      { status: 500 }
    );
  }
}
