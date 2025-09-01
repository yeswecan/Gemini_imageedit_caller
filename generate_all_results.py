#!/usr/bin/env python3
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from image_processor import ImageProcessor

# Initialize processor
processor = ImageProcessor()

def process_image_pair(character_path, selfie_path, results_dir):
    """Process a single character-selfie pair"""
    output_name = f"{character_path.stem}_{selfie_path.stem}_result.png"
    output_path = results_dir / output_name
    
    print(f"Processing: {character_path.name} + {selfie_path.name}")
    
    # Use the image processor with alignment
    result = processor.process_images(
        template_path=character_path,
        selfie_path=selfie_path,
        output_path=output_path,
        align=True
    )
    
    # Convert to expected format for table generation
    table_result = {
        "character": character_path.name,
        "selfie": selfie_path.name,
        "output": output_name if result['success'] else None,
        "time": result['generation_time'],
        "total_time": result['total_time'],
        "retries": result['retries'],
        "error": result.get('error', result.get('alignment_error')),
        "success": result['success'],
        "alignment_failed": 'alignment_error' in result
    }
    
    if result['success']:
        print(f"✓ Completed: {output_name} (gen: {result['generation_time']:.2f}s, total: {result['total_time']:.2f}s, retries: {result['retries']})")
    else:
        print(f"✗ Failed: {character_path.name} + {selfie_path.name} - {result.get('error', 'Unknown error')}")
    
    return table_result

def generate_markdown_table(results, characters, selfies):
    """Generate markdown table from results"""
    md_content = "# Image Generation Results\n\n"
    md_content += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"Model: {processor.model}\n\n"
    md_content += f"Prompt: {processor.prompt}\n\n"
    
    # Create results lookup
    results_dict = {}
    for r in results:
        key = (r["character"], r["selfie"])
        results_dict[key] = r
    
    # Generate table header with selfie images
    md_content += "| Character |"
    for selfie in selfies:
        # Show the selfie image in the header
        selfie_img = f"![{selfie.name}](selfies_samples/{selfie.name})"
        md_content += f" {selfie_img}<br>**{selfie.stem}** |"
    md_content += "\n"
    
    md_content += "|-----------|"
    for _ in selfies:
        md_content += ":---:|"  # Center align columns
    md_content += "\n"
    
    # Generate table rows with character images
    for character in characters:
        # Show the character image in the first column
        char_img = f"![{character.name}](characters/{character.name})"
        md_content += f"| {char_img}<br>**{character.stem}** |"
        
        for selfie in selfies:
            key = (character.name, selfie.name)
            result = results_dict.get(key, {})
            
            if result.get("success"):
                # Success - show image and time
                # Use relative path for GitHub compatibility
                img_path = f"results/{result['output']}"
                cell_content = f"![{result['output']}]({img_path})<br>{result['time']:.2f}s"
            else:
                # Failure - show error
                error = result.get("error", "Unknown error")
                # Don't truncate errors anymore to see full response
                # Escape HTML characters for markdown
                error = error.replace("<", "&lt;").replace(">", "&gt;")
                # Make JSON more readable by adding line breaks
                if "Full response:" in error or "Response:" in error:
                    error = error.replace('", "', '",<br>"').replace('{', '{<br>').replace('}', '<br>}')
                cell_content = f"❌ Error<br><details><summary>Click to expand</summary><pre>{error}</pre></details><br>{result.get('time', 0):.2f}s"
            
            md_content += f" {cell_content} |"
        
        md_content += "\n"
    
    # Add statistics
    total = len(results)
    successful = sum(1 for r in results if r.get("success"))
    failed = total - successful
    avg_time = sum(r.get("time", 0) for r in results) / total if total > 0 else 0
    
    md_content += f"\n## Statistics\n\n"
    md_content += f"- Total requests: {total}\n"
    md_content += f"- Successful: {successful} ({successful/total*100:.1f}%)\n"
    md_content += f"- Failed: {failed} ({failed/total*100:.1f}%)\n"
    md_content += f"- Average time: {avg_time:.2f}s\n"
    
    return md_content

def main():
    """Process all images and generate markdown table"""
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Get all characters and selfies
    characters_dir = Path("characters")
    selfies_dir = Path("selfies_samples")
    
    characters = sorted(list(characters_dir.glob("*.png")))
    selfies = sorted(list(selfies_dir.glob("*.jpg")))
    
    print(f"Found {len(characters)} characters and {len(selfies)} selfies")
    print(f"Total combinations to process: {len(characters) * len(selfies)}")
    print()
    
    # Process all combinations
    results = []
    tasks = []
    generation_log = {}
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = 3  # Limit concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for character in characters:
            for selfie in selfies:
                future = executor.submit(process_image_pair, character, selfie, results_dir)
                tasks.append(future)
        
        # Collect results as they complete
        for future in as_completed(tasks):
            try:
                result = future.result()
                results.append(result)
                
                # Save to generation log
                log_key = f"{Path(result['character']).stem}_{Path(result['selfie']).stem}"
                generation_log[log_key] = {
                    'time': result['time'],
                    'retries': result['retries'],
                    'success': result['success'],
                    'error': result.get('error')
                }
            except Exception as e:
                print(f"Task failed with exception: {e}")
    
    # Save generation log
    import json
    log_path = Path("generation_log.json")
    with open(log_path, 'w') as f:
        json.dump(generation_log, f, indent=2)
    print(f"Generation log saved to: {log_path}")
    
    # Generate markdown table
    print("\nGenerating markdown table...")
    md_content = generate_markdown_table(results, characters, selfies)
    
    # Save markdown file
    md_path = Path("results_table.md")
    md_path.write_text(md_content)
    print(f"Markdown table saved to: {md_path}")
    
    # Print summary
    successful = sum(1 for r in results if r.get("success"))
    print(f"\nCompleted! {successful}/{len(results)} successful generations")

if __name__ == "__main__":
    main()