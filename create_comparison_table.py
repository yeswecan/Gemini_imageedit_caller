#!/usr/bin/env python3
"""
Create a comparison table showing original vs aligned results.
"""
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def create_comparison_table():
    """Create a visual comparison of alignment results."""
    # Sample a few results to show
    samples = [
        ("Ali_F", "005_Selfie_12"),
        ("Boya_F", "041_Selfie_6"),
        ("Champ_M", "023_Selfie_3"),
        ("Karl_F", "027_Selfie_7")
    ]
    
    # Create comparison images
    comparisons = []
    
    for char_name, selfie_id in samples:
        template_path = Path(f"characters/{char_name}.png")
        original_path = Path(f"results_backup_before_alignment/{char_name}_{selfie_id}_result.png")
        aligned_path = Path(f"results/{char_name}_{selfie_id}_result.png")
        
        if not original_path.exists():
            original_path = aligned_path  # Use aligned if no backup
        
        if template_path.exists() and aligned_path.exists():
            # Load images
            template = Image.open(template_path)
            original = Image.open(original_path)
            aligned = Image.open(aligned_path)
            
            # Resize to consistent height
            height = 400
            template = template.resize((int(template.width * height / template.height), height), Image.Resampling.LANCZOS)
            original = original.resize((int(original.width * height / original.height), height), Image.Resampling.LANCZOS)
            aligned = aligned.resize((int(aligned.width * height / aligned.height), height), Image.Resampling.LANCZOS)
            
            # Create comparison strip
            strip_width = template.width + original.width + aligned.width + 40
            strip = Image.new('RGB', (strip_width, height + 50), 'white')
            
            # Paste images
            strip.paste(template, (10, 30))
            strip.paste(original, (template.width + 20, 30))
            strip.paste(aligned, (template.width + original.width + 30, 30))
            
            # Add labels
            draw = ImageDraw.Draw(strip)
            draw.text((10, 5), "Template", fill='black')
            draw.text((template.width + 20, 5), "Generated", fill='black')
            draw.text((template.width + original.width + 30, 5), "Aligned", fill='black')
            
            comparisons.append(strip)
    
    # Stack comparisons vertically
    if comparisons:
        total_height = sum(img.height for img in comparisons) + 50
        max_width = max(img.width for img in comparisons)
        
        final = Image.new('RGB', (max_width, total_height), 'white')
        y = 10
        
        # Add title
        draw = ImageDraw.Draw(final)
        draw.text((10, y), "Face Alignment Comparison", fill='black')
        y += 30
        
        for comp in comparisons:
            final.paste(comp, (0, y))
            y += comp.height
        
        output_path = Path("alignment_comparison_table.png")
        final.save(output_path)
        print(f"Saved comparison to: {output_path}")
    
    # Also check statistics
    results_dir = Path("results")
    backup_dir = Path("results_backup_before_alignment")
    
    if backup_dir.exists():
        original_sizes = []
        aligned_sizes = []
        
        for result in results_dir.glob("*.png"):
            backup = backup_dir / result.name
            if backup.exists():
                original_sizes.append(backup.stat().st_size)
                aligned_sizes.append(result.stat().st_size)
        
        if original_sizes:
            avg_original = sum(original_sizes) / len(original_sizes) / 1024
            avg_aligned = sum(aligned_sizes) / len(aligned_sizes) / 1024
            
            print(f"\nFile size statistics:")
            print(f"Average original size: {avg_original:.1f} KB")
            print(f"Average aligned size: {avg_aligned:.1f} KB")
            print(f"Size change: {(avg_aligned - avg_original) / avg_original * 100:.1f}%")

if __name__ == "__main__":
    create_comparison_table()