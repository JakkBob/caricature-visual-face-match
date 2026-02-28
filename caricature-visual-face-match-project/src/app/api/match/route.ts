/**
 * Cross-Modal Matching API
 * POST /api/match
 * 
 * Proxies requests to Python backend service
 */

import { NextRequest, NextResponse } from 'next/server';
import { readFile, readdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';

// Python service URL
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';

// Supported image formats
const SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];

/**
 * Load images from directory as base64
 */
async function loadImagesFromDirectory(dirPath: string): Promise<{ images: string[]; ids: string[] }> {
  const images: string[] = [];
  const ids: string[] = [];

  if (!existsSync(dirPath)) {
    return { images, ids };
  }

  const files = await readdir(dirPath);

  for (const file of files) {
    const ext = file.split('.').pop()?.toLowerCase();
    if (!ext || !SUPPORTED_FORMATS.includes(ext)) continue;

    const filePath = path.join(dirPath, file);
    const imageBuffer = await readFile(filePath);
    const base64 = imageBuffer.toString('base64');

    images.push(base64);
    ids.push(file);
  }

  return { images, ids };
}

/**
 * Execute cross-modal matching
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
      topK = 10,
      threshold = 0.5,
    } = body;

    // Parameter validation
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

    // Load target images from directory
    const targetDir = path.join(
      process.cwd(),
      'data',
      targetModality === 'face' ? 'faces' : 'caricatures'
    );

    const { images: galleryImages, ids: galleryIds } = await loadImagesFromDirectory(targetDir);

    if (galleryImages.length === 0) {
      return NextResponse.json(
        { success: false, message: 'No target images found in database' },
        { status: 400 }
      );
    }

    // Prepare query image
    let queryImageData: string;
    if (queryImage.startsWith('data:')) {
      // Base64 format with data URL prefix
      queryImageData = queryImage.split(',')[1] || queryImage;
    } else if (queryImage.startsWith('/')) {
      // File path
      const fileBuffer = await readFile(queryImage);
      queryImageData = fileBuffer.toString('base64');
    } else {
      // Assume already Base64
      queryImageData = queryImage;
    }

    // Call Python service
    const response = await fetch(`${PYTHON_SERVICE_URL}/match`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query_image: queryImageData,
        gallery_images: galleryImages,
        gallery_ids: galleryIds,
        top_k: topK,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { success: false, message: `Python service error: ${error}` },
        { status: response.status }
      );
    }

    const result = await response.json();

    // Update is_match based on threshold
    if (result.matches) {
      result.matches = result.matches.map((m: { similarity: number }) => ({
        ...m,
        isMatch: m.similarity >= threshold,
      }));
    }

    return NextResponse.json({
      success: result.success,
      data: result,
      message: result.message,
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
 * Compute similarity between two images
 * PUT /api/match
 * 
 * Body: {
 *   image1: string (Base64),
 *   image2: string (Base64)
 * }
 */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const { image1, image2 } = body;

    if (!image1 || !image2) {
      return NextResponse.json(
        { success: false, message: 'Both images are required' },
        { status: 400 }
      );
    }

    // Prepare images
    const prepareImage = (image: string): string => {
      if (image.startsWith('data:')) {
        return image.split(',')[1] || image;
      }
      return image;
    };

    const img1Data = prepareImage(image1);
    const img2Data = prepareImage(image2);

    // Call Python service
    const response = await fetch(`${PYTHON_SERVICE_URL}/similarity`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image1: img1Data,
        image2: img2Data,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { success: false, message: `Python service error: ${error}` },
        { status: response.status }
      );
    }

    const result = await response.json();

    return NextResponse.json({
      success: result.success,
      data: result.similarity !== undefined ? { similarity: result.similarity } : undefined,
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
