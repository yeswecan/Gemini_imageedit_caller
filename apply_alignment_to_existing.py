#!/usr/bin/env python3
"""
Apply face alignment to existing generated results.
"""
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from face_alignment_unified import align_generated_image
import shutil

def process_existing_result(character_path, result_path, aligned_dir):
    """Apply alignment to an existing result."""
    output_name = result_path.name
    output_path = aligned_dir / output_name
    
    print(f"Aligning: {output_name}")
    
    # Apply alignment
    alignment_result = align_generated_image(
        generated_path=result_path,
        template_path=character_path,
        output_path=output_path
    )
    
    if alignment_result['success']:
        print(f"✓ Aligned: {output_name} (scale: {alignment_result.get('scale', 1.0):.3f})")
        return True
    else:
        # If alignment fails, keep original; avoid copying onto itself
        print(f"✗ Alignment failed: {output_name} - {alignment_result['error']}")
        if result_path.resolve() != output_path.resolve():
            shutil.copy2(result_path, output_path)
        return False

def main():
    """Apply alignment to all existing results."""
    characters_dir = Path("characters")
    # Prefer current results to ensure full coverage of combinations
    results_dir = Path("results")
    aligned_dir = Path("results")
    
    # Get all results
    results = list(results_dir.glob("*.png"))
    print(f"Found {len(results)} results to align")
    
    # Process each result
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for result_path in results:
            # Extract character name from result filename
            # Format: CharacterName_XXX_Selfie_YY_result.png
            # Find the last occurrence of "_result" and work backwards
            filename = result_path.stem
            if filename.endswith('_result'):
                # Remove '_result' suffix
                base_name = filename[:-7]  # Remove '_result'
                
                # Find the selfie pattern (e.g., "005_Selfie_12")
                import re
                selfie_pattern = r'_\d{3}_Selfie_\d+$'
                match = re.search(selfie_pattern, base_name)
                
                if match:
                    # Character name is everything before the selfie pattern
                    character_name = base_name[:match.start()]
                    character_path = characters_dir / f"{character_name}.png"
                else:
                    print(f"Warning: Could not parse filename: {result_path.name}")
                    character_path = None
                
                if character_path and character_path.exists():
                    future = executor.submit(
                        process_existing_result,
                        character_path,
                        result_path,
                        aligned_dir
                    )
                    futures.append(future)
                else:
                    print(f"Warning: Template not found for {result_path.name}")
                    # Copy original
                    shutil.copy2(result_path, aligned_dir / result_path.name)
        
        # Collect results
        for future in as_completed(futures):
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing: {e}")
                failed += 1
    
    print(f"\nAlignment complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(results)}")

if __name__ == "__main__":
    main()