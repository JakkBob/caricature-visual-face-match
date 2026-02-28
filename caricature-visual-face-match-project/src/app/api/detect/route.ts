/**
 * Face Detection API
 * POST /api/detect
 * 
 * Proxies requests to Python backend service
 */

import { NextRequest, NextResponse } from 'next/server';

// Python service URL
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';

/**
 * Detect faces in image
 * POST /api/detect
 * 
 * Body: {
 *   image: string (Base64),
 *   alignSize?: number
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { image, alignSize = 224 } = body;

    if (!image) {
      return NextResponse.json(
        { success: false, message: 'Image is required' },
        { status: 400 }
      );
    }

    // Call Python service
    const response = await fetch(`${PYTHON_SERVICE_URL}/detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: image,
        align_size: alignSize,
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
    console.error('Detect error:', error);
    return NextResponse.json(
      { success: false, message: `Detection failed: ${error}` },
      { status: 500 }
    );
  }
}
