#!/usr/bin/env python3
import time
from pathlib import Path

# Read prompt
prompt_path = Path("prompt.md")
PROMPT = prompt_path.read_text().strip()
MODEL = "google/gemini-2.5-flash-image-preview"

def generate_simple_markdown_table():
    """Generate simplified markdown table with just images or X emoji"""
    # Get all characters and selfies
    characters_dir = Path("characters")
    selfies_dir = Path("selfies_samples")
    results_dir = Path("results")
    
    characters = sorted(list(characters_dir.glob("*.png")))
    selfies = sorted(list(selfies_dir.glob("*.jpg")))[:8]  # Use only first 8 selfies
    
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
                # Success - show image only
                img_path = f"results/{output_name}"
                cell_content = f"![{output_name}]({img_path})"
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