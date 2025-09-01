#!/usr/bin/env python3
"""
REST API server for AI-powered face swapping with automatic alignment.
Accepts template illustration and selfie photo, returns aligned face-swapped result.
"""
import os
import base64
import requests
import json
import time
import tempfile
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import io
from image_processor_no_align import ImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize image processor
processor = ImageProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': processor.model,
        'version': '1.0.0'
    })

@app.route('/swap_face', methods=['POST'])
def swap_face():
    """
    Main endpoint for face swapping with alignment.
    Expects multipart/form-data with:
    - template: Template illustration image file
    - selfie: Selfie photo image file
    
    Returns:
    - Aligned face-swapped image
    """
    start_time = time.time()
    
    # Validate request
    if 'template' not in request.files or 'selfie' not in request.files:
        return jsonify({
            'error': 'Missing required files. Please provide both template and selfie images.'
        }), 400
    
    template_file = request.files['template']
    selfie_file = request.files['selfie']
    
    if template_file.filename == '' or selfie_file.filename == '':
        return jsonify({
            'error': 'Empty filename provided'
        }), 400
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save uploaded files
        template_path = temp_path / f"template_{template_file.filename}"
        selfie_path = temp_path / f"selfie_{selfie_file.filename}"
        generated_path = temp_path / "generated.png"
        aligned_path = temp_path / "aligned.png"
        
        template_file.save(str(template_path))
        selfie_file.save(str(selfie_path))
        
        logger.info(f"Processing: {template_file.filename} + {selfie_file.filename}")
        
        # Process images using the image processor
        result = processor.process_images(
            template_path=template_path,
            selfie_path=selfie_path,
            output_path=aligned_path,
            align=True
        )
        
        if not result['success']:
            logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")
            return jsonify({
                'error': result.get('error', 'Processing failed'),
                'generation_time': result['generation_time'],
                'retries': result['retries']
            }), 500
        
        # Success - return aligned image
        logger.info(f"Success: Generated and aligned in {result['total_time']:.2f}s")
        
        # Prepare response headers
        headers = {
            'X-Generation-Time': str(result['generation_time']),
            'X-Alignment-Status': 'failed' if result.get('alignment_error') else 'success',
            'X-Total-Time': str(result['total_time']),
            'X-Retries': str(result['retries'])
        }
        
        if 'scale' in result:
            headers['X-Scale-Factor'] = str(result['scale'])
        if 'angle' in result:
            headers['X-Rotation-Angle'] = str(result['angle'])
        if result.get('alignment_error'):
            headers['X-Alignment-Error'] = result['alignment_error']
        
        return send_file(
            str(aligned_path),
            mimetype='image/png',
            as_attachment=False,
            download_name='aligned_result.png'
        ), 200, headers

@app.route('/test_alignment', methods=['POST'])
def test_alignment():
    """
    Test endpoint for alignment only (no generation).
    Expects multipart/form-data with:
    - template: Template illustration image file
    - generated: Pre-generated image file to align
    
    Returns:
    - Aligned image
    """
    if 'template' not in request.files or 'generated' not in request.files:
        return jsonify({
            'error': 'Missing required files. Please provide both template and generated images.'
        }), 400
    
    template_file = request.files['template']
    generated_file = request.files['generated']
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        template_path = temp_path / f"template_{template_file.filename}"
        generated_path = temp_path / f"generated_{generated_file.filename}"
        aligned_path = temp_path / "aligned.png"
        
        template_file.save(str(template_path))
        generated_file.save(str(generated_path))
        
        # Align image using face_alignment module directly
        from face_alignment import align_generated_image
        alignment_result = align_generated_image(
            generated_path,
            template_path,
            aligned_path
        )
        
        if not alignment_result['success']:
            return jsonify({
                'error': f'Alignment failed: {alignment_result["error"]}'
            }), 500
        
        return send_file(
            str(aligned_path),
            mimetype='image/png',
            as_attachment=False,
            download_name='aligned_test.png'
        )

@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler"""
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)