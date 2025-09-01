#!/usr/bin/env python3
import time
import json
from pathlib import Path

# Read prompt
prompt_path = Path("prompt.md")
PROMPT = prompt_path.read_text().strip()
MODEL = "google/gemini-2.5-flash-image-preview"

def read_generation_log():
    """Read generation log if it exists"""
    log_path = Path("generation_log.json")
    if log_path.exists():
        with open(log_path, 'r') as f:
            return json.load(f)
    return {}

def generate_simple_markdown_table():
    """Generate simplified markdown table with images, times, and retry counts"""
    # Get all characters and selfies
    characters_dir = Path("characters")
    selfies_dir = Path("selfies_samples")
    results_dir = Path("results")
    
    characters = sorted(list(characters_dir.glob("*.png")))
    selfies = sorted(list(selfies_dir.glob("*.jpg")))[:8]  # Use only first 8 selfies
    
    # Read generation log for timing and retry info
    gen_log = read_generation_log()
    
    # Generate markdown content
    md_content = "# Image Generation Results\n\n"
    md_content += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"Model: {MODEL}\n\n"
    md_content += f"Prompt: {PROMPT}\n\n"
    
    # Generate table header with selfie images
    md_content += "| Character |"
    for selfie in selfies:
        # Show the selfie image in the header
        selfie_img = f"![{selfie.name}](selfies_samples/{selfie.name})"
        md_content += f" {selfie_img} |"
    md_content += "\n"
    
    md_content += "|:---:|"  # Center align first column
    for _ in selfies:
        md_content += ":---:|"  # Center align all columns
    md_content += "\n"
    
    # Generate table rows with character images
    for character in characters:
        # Show the character image in the first column
        char_img = f"![{character.name}](characters/{character.name})"
        md_content += f"| {char_img} |"
        
        for selfie in selfies:
            output_name = f"{character.stem}_{selfie.stem}_result.png"
            output_path = results_dir / output_name
            
            if output_path.exists():
                # Success - show image with time and retries if available
                img_path = f"results/{output_name}"
                cell_content = f"![{output_name}]({img_path})"
                
                # Add timing info if available from log
                log_key = f"{character.stem}_{selfie.stem}"
                if log_key in gen_log:
                    gen_time = gen_log[log_key].get('time', 0)
                    retries = gen_log[log_key].get('retries', 0)
                    cell_content += f"<br>{gen_time:.1f}s"
                    if retries > 0:
                        cell_content += f" ({retries} retry)"
            else:
                # Failure - show X emoji
                cell_content = "❌"
            
            md_content += f" {cell_content} |"
        
        md_content += "\n"
    
    # Add simple statistics
    total_combinations = len(characters) * len(selfies)
    successful = sum(1 for c in characters for s in selfies 
                    if (results_dir / f"{c.stem}_{s.stem}_result.png").exists())
    
    md_content += f"\n## Statistics\n\n"
    md_content += f"- Total: {total_combinations}\n"
    md_content += f"- Successful: {successful} ({successful/total_combinations*100:.1f}%)\n"
    
    return md_content

def main():
    """Regenerate simplified markdown table"""
    print("Regenerating simplified markdown table...")
    
    md_content = generate_simple_markdown_table()
    
    # Save markdown file
    md_path = Path("results_table.md")
    md_path.write_text(md_content)
    print(f"Simplified markdown table saved to: {md_path}")

if __name__ == "__main__":
    main()