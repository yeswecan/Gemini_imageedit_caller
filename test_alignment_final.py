#!/usr/bin/env python3
"""
Final alignment test with visualization.
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

def create_alignment_visualization():
    """Create a clear visualization of the alignment process."""
    print("Creating alignment visualization...")
    
    # Load images
    template_path = Path("characters/Ali_F.png")
    generated_path = Path("results/Ali_F_005_Selfie_12_result.png")
    
    # Load with PIL for easier manipulation
    template = Image.open(template_path)
    generated = Image.open(generated_path)
    
    print(f"Template size: {template.size}")
    print(f"Generated size: {generated.size}")
    
    # Create a visualization grid
    # We'll show: Original Template | Original Generated | Aligned Result
    
    # For visualization, let's resize generated to match template height
    template_w, template_h = template.size
    gen_w, gen_h = generated.size
    
    # Calculate scale to match heights
    scale = template_h / gen_h
    new_gen_w = int(gen_w * scale)
    
    # Resize generated to match template height
    generated_resized = generated.resize((new_gen_w, template_h), Image.Resampling.LANCZOS)
    
    # Create canvas for side-by-side comparison
    canvas_width = template_w * 3 + 40  # 3 images + spacing
    canvas_height = template_h + 100  # Extra space for labels
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Paste images
    canvas.paste(template, (10, 50))
    canvas.paste(generated_resized, (template_w + 20, 50))
    
    # For the aligned version, center the resized generated on template position
    # This simulates what proper alignment would do
    aligned_canvas = Image.new('RGBA', template.size, (255, 255, 255, 0))
    
    # Center the face
    x_offset = (template_w - new_gen_w) // 2
    if x_offset < 0:
        # Generated is wider, crop it
        crop_x = (-x_offset) // 2
        generated_cropped = generated_resized.crop((crop_x, 0, crop_x + template_w, template_h))
        aligned_canvas.paste(generated_cropped, (0, 0))
    else:
        # Template is wider, center generated
        aligned_canvas.paste(generated_resized, (x_offset, 0))
    
    # Paste aligned result
    canvas.paste(aligned_canvas, (template_w * 2 + 30, 50))
    
    # Add labels
    draw.text((10, 10), "Template", fill='black')
    draw.text((template_w + 20, 10), "Generated (scaled)", fill='black')
    draw.text((template_w * 2 + 30, 10), "Aligned Result", fill='black')
    
    # Add info text
    info_y = template_h + 60
    draw.text((10, info_y), f"Scale factor: {scale:.2f}x", fill='black')
    
    # Save
    output_path = Path("test_alignment_visualization.png")
    canvas.save(output_path)
    print(f"\nSaved visualization to: {output_path}")
    
    # Also create a simple overlay to show alignment
    overlay = Image.new('RGBA', template.size, (255, 255, 255, 255))
    
    # Paste template with transparency
    template_rgba = template.convert('RGBA')
    overlay.paste(template_rgba, (0, 0))
    
    # Paste aligned generated with transparency
    aligned_rgba = aligned_canvas.convert('RGBA')
    # Make it semi-transparent to see overlay
    aligned_data = aligned_rgba.getdata()
    new_data = []
    for item in aligned_data:
        # Change alpha to 128 (50% transparent) for non-transparent pixels
        if item[3] > 0:
            new_data.append((item[0], item[1], item[2], 128))
        else:
            new_data.append(item)
    aligned_rgba.putdata(new_data)
    
    overlay = Image.alpha_composite(overlay, aligned_rgba)
    
    overlay_path = Path("test_alignment_overlay.png")
    overlay.save(overlay_path)
    print(f"Saved overlay to: {overlay_path}")
    
    print("\nAlignment test complete!")
    print("The visualization shows how the generated face would be")
    print("scaled and positioned to match the template.")

if __name__ == "__main__":
    create_alignment_visualization()