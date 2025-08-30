"""
Add character positions to existing segmentation results.
"""

import json
import os


def add_char_positions_to_segmentation(input_file: str, output_file: str = None):
    """
    Add character start/end positions to existing segmentation.
    
    Args:
        input_file: Path to existing segmentation JSON
        output_file: Path to save updated segmentation (defaults to overwriting input)
    """
    if output_file is None:
        output_file = input_file
    
    # Load existing segmentation
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sections = data['sections']
    char_position = 0
    
    print(f"Adding character positions to {len(sections)} sections...")
    
    for i, section in enumerate(sections):
        # Calculate character positions
        section_text = section['text']
        section_char_start = char_position
        section_char_end = char_position + len(section_text)
        
        # Add character positions
        section['char_start'] = section_char_start
        section['char_end'] = section_char_end
        
        # Safe printing with ASCII fallback
        title_safe = section['title'][:50].encode('ascii', 'replace').decode('ascii')
        print(f"Section {i}: '{title_safe}...' -> chars {section_char_start}-{section_char_end}")
        
        # Update character position for next section (add newline between sections)
        char_position = section_char_end + 1
    
    # Update timestamp
    from datetime import datetime
    data['timestamp'] = datetime.now().isoformat()
    data['char_positions_added'] = True
    
    # Save updated segmentation
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Updated segmentation saved to: {output_file}")
    print(f"Total document length: {char_position} characters")


if __name__ == "__main__":
    # Update the existing segmentation file
    input_file = r"C:\Source\vectorless\segmentation_results\LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR_AGREEMEN_cached.json"
    
    if os.path.exists(input_file):
        add_char_positions_to_segmentation(input_file)
    else:
        print(f"File not found: {input_file}")