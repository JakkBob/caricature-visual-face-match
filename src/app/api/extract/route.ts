/**
 * Feature Extraction API
 * POST /api/extract
 * 
 * Proxies requests to Python backend service
 */

import { NextRequest, NextResponse } from 'next/server';

// Python service URL
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';

/**
 * Extract features from image
 * POST /api/extract
 * 
 * Body: {
 *   image: string (Base64),
 *   modality: 'face' | 'caricature'
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { image, modality = 'face' } = body;

    if (!image) {
      return NextResponse.json(
        { success: false, message: 'Image is required' },
        { status: 400 }
      );
    }

    if (!['face', 'caricature'].includes(modality)) {
      return NextResponse.json(
        { success: false, message: 'Invalid modality. Must be "face" or "caricature"' },
        { status: 400 }
      );
    }

    // Call Python service
    const response = await fetch(`${PYTHON_SERVICE_URL}/extract`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: image,
        modality: modality,
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
    return NextResponse.json(result);
  } catch (error) {
    console.error('Extract error:', error);
    return NextResponse.json(
      { success: false, message: `Feature extraction failed: ${error}` },
      { status: 500 }
    );
  }
}
