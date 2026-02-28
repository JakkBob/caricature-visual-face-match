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
 * Check if a string is valid Base64
 */
function isBase64(str: string): boolean {
  // Check if it's a data URL
  if (str.startsWith('data:')) {
    return true;
  }
  
  // Check if it's a valid Base64 string (only contains Base64 characters)
  // Base64 charset: A-Z, a-z, 0-9, +, /, = (padding)
  const base64Regex = /^[A-Za-z0-9+/]+=*$/;
  
  // Also check if the string is long enough to be image data
  // and doesn't look like a file path
  if (str.length > 100 && base64Regex.test(str.replace(/\s/g, ''))) {
    return true;
  }
  
  return false;
}

/**
 * Check if a string is a valid file path
 */
function isFilePath(str: string): boolean {
  // Must start with / or a drive letter (Windows) or ./
  // and must not contain Base64-only characters in sequence
  if (str.startsWith('data:')) {
    return false;
  }
  
  // Check for Windows path (e.g., C:\, D:\)
  if (/^[A-Za-z]:[/\\]/.test(str)) {
    return true;
  }
  
  // Check for Unix absolute path
  if (str.startsWith('/')) {
    // But not if it looks like Base64 (long string with only Base64 chars)
    if (str.length > 200 && /^[A-Za-z0-9+/]+=*$/.test(str.substring(1))) {
      return false;
    }
    return true;
  }
  
  // Check for relative path
  if (str.startsWith('./') || str.startsWith('../')) {
    return true;
  }
  
  return false;
}

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
    } else if (isFilePath(queryImage)) {
      // File path - read file
      try {
        const fileBuffer = await readFile(queryImage);
        queryImageData = fileBuffer.toString('base64');
      } catch (fileError) {
        return NextResponse.json(
          { success: false, message: `Failed to read image file: ${queryImage}` },
          { status: 400 }
        );
      }
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
        query_modality: queryModality,
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
    console.log('[Match API] Python service result:', JSON.stringify(result, null, 2));

    // Update is_match based on threshold
    if (result.matches) {
      result.matches = result.matches.map((m: { similarity: number }) => ({
        ...m,
        isMatch: m.similarity >= threshold,
      }));
    }

    console.log('[Match API] Returning to client:', JSON.stringify({
      success: result.success,
      data: result,
      message: result.message,
    }, null, 2));

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
