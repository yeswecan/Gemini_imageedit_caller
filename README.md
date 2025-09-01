# AI Face Swap API Server

Automated face transfer API server using Google's Gemini 2.5 Flash Image Preview model. The system transfers facial features and hairstyles from selfie photos onto illustrated character templates with automatic alignment using InsightFace landmark detection.

## Overview

This project provides a REST API server for AI-powered face swapping with automatic face alignment. The core functionality:
- Accepts a template illustration and a selfie photo
- Generates a face swap using Gemini API
- Automatically aligns the result to match the template's face position/scale
- Returns the aligned image ready for use

**Success Rate**: 145/160 (90.6%) in batch testing - see [results_table.md](results_table.md)

## How It Works

The server processes requests through a three-stage pipeline. First, it sends the template illustration and selfie to Google's Gemini model via OpenRouter API to generate the face swap. Then, InsightFace detects facial landmarks (eyes and mouth centers) in both the template and generated images. Finally, the system calculates the required scale, rotation, and translation to align the generated face with the template's face position, ensuring perfect placement in the final output.

**Note**: Face alignment works best with photorealistic images. For stylized illustrations, InsightFace may have difficulty detecting landmarks, resulting in alignment failures. In such cases, the system returns the unaligned generated image.

## API Endpoints

### `POST /swap_face`
Main endpoint for face swapping with alignment.

**Request**: `multipart/form-data`
- `template`: Template illustration image file (PNG/JPG)
- `selfie`: Selfie photo image file (PNG/JPG)

**Response**: Aligned face-swapped image (PNG)

**Response Headers**:
- `X-Generation-Time`: Time taken for AI generation
- `X-Alignment-Status`: success/failed
- `X-Total-Time`: Total processing time
- `X-Retries`: Number of API retries

### `GET /health`
Health check endpoint returning server status.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenRouter API key in `.env`:
```
OPENROUTER_API_KEY=your_key_here
```

3. Start the server:
```bash
python server.py
```

4. Test with curl:
```bash
curl -X POST http://localhost:5000/swap_face \
  -F "template=@characters/Ali_F.png" \
  -F "selfie=@selfies_samples/005_Selfie_12.jpg" \
  --output result.png
```

## Batch Processing

For testing or batch processing, use:
```bash
python generate_all_results.py
```

This processes all character-selfie combinations and generates a visual results table with timing and retry information.