# Gemini Image Edit Caller

Automated face and hair transfer using Google's Gemini 2.5 Flash Image Preview model via OpenRouter API. The project applies facial features and hairstyles from real selfie photos onto illustrated character designs.

## Overview

- **Input**: 20 character illustrations × 8 selfie photos = 160 combinations
- **Success Rate**: 145 successful generations (90.6%)
- **Results**: View the complete grid in [results_table.md](results_table.md)

## How It Works

The system takes two images: a character illustration and a selfie photo. Using the Gemini API, it transfers the face and hairstyle from the selfie onto the character while preserving the original pose, outfit, background, and art style. Each generation takes about 18 seconds, with automatic retry logic and rate limiting for reliable batch processing.

## Repository Structure

```
├── characters/          # 20 source character illustrations
├── selfies_samples/     # 8 source selfie photos
├── results/            # 145 generated images
├── results_table.md    # Visual grid of all transformations
├── generate_all_results.py  # Main batch processing script
└── test_single_image.py     # Single pair testing script
```

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Add your OpenRouter API key to `.env` file
3. Run: `python generate_all_results.py`

## Example Results

![Results Preview](results_table.md)

Failed generations (marked with ❌) are typically due to content filtering.