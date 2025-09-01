# Gemini Image Edit Caller

This repository demonstrates automated face and hair transfer using Google's Gemini 2.5 Flash Image Preview model via OpenRouter API. It systematically applies facial features and hairstyles from real selfie photos onto illustrated character designs.

## Overview

The project tests the capabilities of the `google/gemini-2.5-flash-image-preview` model by:
- Taking illustrated character designs (20 different characters)
- Applying faces and hairstyles from real selfie photos (8 different selfies)
- Generating 160 unique combinations while preserving the original pose, outfit, background, and art style

## Results

View the complete results grid: [results_table.md](results_table.md)

### Success Rate
- **Total combinations**: 160
- **Successful generations**: 145 (90.6%)
- **Failed generations**: 15 (9.4%) - mostly due to content filtering

## Repository Structure

```
├── characters/          # 20 source character illustrations (PNG)
├── selfies_samples/     # 8 source selfie photos (JPG)
├── results/            # 145 generated result images
├── results_table.md    # Visual grid showing all transformations
├── prompt.md           # The prompt used for image generation
├── test_single_image.py     # Script to test a single image pair
├── generate_all_results.py  # Batch processing script for all combinations
└── regenerate_simple_table.py # Script to regenerate the results table
```

## How It Works

1. **Input**: Each generation takes two images:
   - A character illustration (from `characters/`)
   - A selfie photo (from `selfies_samples/`)

2. **Prompt**: The model is instructed to:
   ```
   Change the person on the illustration to the person on the photo. 
   Change the hair on illustration to the hair on the photo. 
   Keep the pose and outfit intact.
   Keep the background intact. Keep the style intact. 
   Change the hair. Change the face.
   ```

3. **Output**: A new image combining the selfie's facial features and hairstyle with the character's pose, outfit, and artistic style.

## Technical Details

### API Integration
- **Model**: google/gemini-2.5-flash-image-preview
- **Provider**: OpenRouter
- **Endpoint**: https://openrouter.ai/api/v1/chat/completions
- **Authentication**: Bearer token via API key

### Features
- Automatic retry logic (up to 3 attempts per image)
- Rate limiting (max 2 requests per second)
- Error handling for content filtering
- Progress tracking and timing metrics
- Parallel processing with thread pooling

## Usage

### Prerequisites
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file with your OpenRouter API key:
```
OPENROUTER_API_KEY=your_api_key_here
```

### Running the Scripts

**Test single image pair:**
```bash
python test_single_image.py
```

**Process all combinations:**
```bash
python generate_all_results.py
```

**Regenerate results table:**
```bash
python regenerate_simple_table.py
```

## Example Results

The results table shows a grid where:
- **Rows**: Different character illustrations
- **Columns**: Different selfie photos  
- **Cells**: Generated images combining the character with the selfie's face/hair

Failed generations are marked with ❌ (typically due to content filtering).

## Limitations

- Some combinations trigger content filters (PROHIBITED_CONTENT)
- Processing time averages ~18 seconds per image
- API rate limits may affect batch processing speed
- Results quality varies based on the compatibility of source images

## License

This project is for demonstration and research purposes. The character illustrations and generated images are subject to their respective licenses and usage terms.