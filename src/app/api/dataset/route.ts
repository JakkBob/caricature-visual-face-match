/**
 * 数据集管理 API
 * GET/POST/DELETE /api/dataset
 */

import { NextRequest, NextResponse } from 'next/server';
import { readdir, stat, unlink, rmdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import ModelConfig from '../../../../config/model.config';

/**
 * 获取数据集信息
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const modality = searchParams.get('modality') as 'face' | 'caricature' | null;

    const getDataInfo = async (dirPath: string, modalityType: string) => {
      if (!existsSync(dirPath)) {
        return { count: 0, files: [], totalSize: 0 };
      }

      const files = await readdir(dirPath);
      const validFiles = [];
      let totalSize = 0;

      for (const file of files) {
        const ext = file.split('.').pop()?.toLowerCase();
        if (!ext || !ModelConfig.dataProcessing.supportedFormats.includes(ext)) continue;

        const filePath = path.join(dirPath, file);
        const fileStat = await stat(filePath);
        
        validFiles.push({
          name: file,
          path: filePath,
          size: fileStat.size,
          modified: fileStat.mtime,
        });
        totalSize += fileStat.size;
      }

      return {
        count: validFiles.length,
        files: validFiles,
        totalSize,
      };
    };

    const facesDir = path.join(process.cwd(), 'data', 'faces');
    const caricaturesDir = path.join(process.cwd(), 'data', 'caricatures');

    if (modality === 'face') {
      const faceInfo = await getDataInfo(facesDir, 'face');
      return NextResponse.json({
        success: true,
        data: { modality: 'face', ...faceInfo },
      });
    }

    if (modality === 'caricature') {
      const caricatureInfo = await getDataInfo(caricaturesDir, 'caricature');
      return NextResponse.json({
        success: true,
        data: { modality: 'caricature', ...caricatureInfo },
      });
    }

    // 返回所有数据集信息
    const [faceInfo, caricatureInfo] = await Promise.all([
      getDataInfo(facesDir, 'face'),
      getDataInfo(caricaturesDir, 'caricature'),
    ]);

    return NextResponse.json({
      success: true,
      data: {
        faces: { modality: 'face', ...faceInfo },
        caricatures: { modality: 'caricature', ...caricatureInfo },
        total: {
          count: faceInfo.count + caricatureInfo.count,
          size: faceInfo.totalSize + caricatureInfo.totalSize,
        },
      },
    });
  } catch (error) {
    console.error('Get dataset error:', error);
    return NextResponse.json(
      { success: false, message: `Failed to get dataset info: ${error}` },
      { status: 500 }
    );
  }
}

/**
 * 删除数据集中的文件
 */
export async function DELETE(request: NextRequest) {
  try {
    const body = await request.json();
    const { files } = body as { files: string[] };

    if (!files || files.length === 0) {
      return NextResponse.json(
        { success: false, message: 'No files specified' },
        { status: 400 }
      );
    }

    const deletedFiles: string[] = [];
    const failedFiles: { file: string; error: string }[] = [];

    for (const file of files) {
      try {
        // 安全检查：确保文件在data目录下
        const normalizedPath = path.normalize(file);
        const dataDir = path.join(process.cwd(), 'data');
        
        if (!normalizedPath.startsWith(dataDir)) {
          failedFiles.push({ file, error: 'Invalid file path' });
          continue;
        }

        if (existsSync(normalizedPath)) {
          await unlink(normalizedPath);
          deletedFiles.push(file);
        }
      } catch (error) {
        failedFiles.push({ file, error: String(error) });
      }
    }

    return NextResponse.json({
      success: true,
      data: {
        deleted: deletedFiles.length,
        failed: failedFiles.length,
        deletedFiles,
        failedFiles,
      },
    });
  } catch (error) {
    console.error('Delete files error:', error);
    return NextResponse.json(
      { success: false, message: `Failed to delete files: ${error}` },
      { status: 500 }
    );
  }
}
