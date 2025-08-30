"""
Legal contract segmentation specialized for CUAD dataset.
Focus on clean contract segmentation only.
"""

import json
import os
from typing import List, Dict, Any, Optional
from .segmentation import segment_document, save_segmentation_to_json, load_segmentation_from_json


class CUADSegmenter:
    """Specialized segmenter for CUAD legal contracts."""
    
    def __init__(self, sample_data_path: str = "sample_dataset/sample_cuad.json"):
        """
        Initialize with CUAD sample data.
        
        Args:
            sample_data_path: Path to the CUAD sample JSON file
        """
        self.sample_data_path = sample_data_path
        self.sample_data = None
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load the CUAD sample data."""
        try:
            with open(self.sample_data_path, 'r') as f:
                self.sample_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Sample data not found at {self.sample_data_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {self.sample_data_path}")
    
    def get_contract_text(self) -> str:
        """Extract the contract text from CUAD sample data."""
        if not self.sample_data or not self.sample_data.get('data'):
            raise ValueError("No data found in sample file")
        
        # Get the first (and only) document
        document = self.sample_data['data'][0]
        
        # Extract the context (contract text) from the first paragraph
        if not document.get('paragraphs') or not document['paragraphs'][0].get('context'):
            raise ValueError("No contract context found in sample data")
        
        return document['paragraphs'][0]['context']
    
    def _get_cache_filepath(self) -> str:
        """Get the filepath for cached segmentation results."""
        contract_title = self.sample_data['data'][0]['title']
        # Clean filename - remove special characters
        clean_title = "".join(c for c in contract_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title.replace(' ', '_')[:50]  # Limit length
        return f"segmentation_results/{clean_title}_cached.json"
    
    def segment_contract(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Segment the contract and return structured data.
        
        Args:
            use_cache: Whether to use cached results if available
        
        Returns:
            Dictionary containing:
            - contract_title: Title of the contract
            - sections: List of segmented sections
            - total_sections: Number of sections
            - cached: Whether results were loaded from cache
        """
        contract_title = self.sample_data['data'][0]['title']
        cache_filepath = self._get_cache_filepath()
        
        # Try to load from cache first
        if use_cache and os.path.exists(cache_filepath):
            try:
                print(f"Loading cached segmentation from: {cache_filepath}")
                cached_data = load_segmentation_from_json(cache_filepath)
                return {
                    "contract_title": cached_data["contract_title"],
                    "sections": cached_data["sections"],
                    "total_sections": cached_data["total_sections"],
                    "cached": True
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Proceeding with new segmentation...")
        
        # Perform new segmentation
        print("Performing new contract segmentation...")
        contract_text = self.get_contract_text()
        
        # Segment the document (don't auto-save since we'll save with proper title)
        sections = segment_document(contract_text, save_results=False)
        
        # Save with proper contract title
        try:
            save_segmentation_to_json(sections, contract_title)
            # Also save as cache file
            os.makedirs("segmentation_results", exist_ok=True)
            result_data = {
                "contract_title": contract_title,
                "timestamp": json.dumps(None),  # Will be replaced by save function
                "total_sections": len(sections),
                "sections": sections
            }
            with open(cache_filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            print(f"Cached results saved to: {cache_filepath}")
        except Exception as e:
            print(f"Warning: Failed to save results: {e}")
        
        return {
            "contract_title": contract_title,
            "sections": sections,
            "total_sections": len(sections),
            "cached": False
        }
    
    def print_segmentation_summary(self) -> None:
        """Print a clean summary of the contract segmentation with enhanced metadata."""
        result = self.segment_contract()
        
        cache_status = " (from cache)" if result.get('cached', False) else " (newly generated)"
        print(f"Contract: {result['contract_title']}")
        print(f"Total Sections: {result['total_sections']}{cache_status}")
        print("\n" + "="*120)
        
        for i, section in enumerate(result['sections']):
            print(f"\nSection {i+1}: {section['title']}")
            print(f"Legal Concept: {section.get('legal_concept', 'N/A')}")
            print(f"Lines: {section['start_index']}-{section['end_index']} ({section['line_count']} lines)")
            
            # Display enhanced summary with encoding handling
            summary = section.get('summary', 'No summary available')
            try:
                print(f"Summary: {summary}")
            except UnicodeEncodeError:
                summary_clean = summary.encode('ascii', 'replace').decode('ascii')
                print(f"Summary: {summary_clean}")
            
            # Display key terms for searchability
            if section.get('key_terms'):
                key_terms_str = ', '.join(section['key_terms'])
                try:
                    print(f"Key Search Terms: {key_terms_str}")
                except UnicodeEncodeError:
                    key_terms_clean = key_terms_str.encode('ascii', 'replace').decode('ascii')
                    print(f"Key Search Terms: {key_terms_clean}")
            
            # Display reasoning if available
            if section.get('reasoning'):
                try:
                    print(f"AI Reasoning: {section['reasoning']}")
                except UnicodeEncodeError:
                    reasoning_clean = section['reasoning'].encode('ascii', 'replace').decode('ascii')
                    print(f"AI Reasoning: {reasoning_clean}")
            
            # Show section text with better formatting
            section_text = section['text'].strip()
            
            # Show first few lines of the section
            lines = section_text.split('\n')[:3]  # Reduced to 3 lines since we have more metadata now
            preview = '\n'.join(lines)
            
            print(f"Content Preview:")
            # Handle encoding issues for Windows
            try:
                print(f"   {preview}")
            except UnicodeEncodeError:
                # Replace problematic Unicode characters
                preview_clean = preview.encode('ascii', 'replace').decode('ascii')
                print(f"   {preview_clean}")
            
            if len(section['text'].split('\n')) > 3:
                print(f"   ... ({len(section['text'].split('\n')) - 3} more lines)")
            
            print("-" * 120)


def demo_segmentation():
    """Demonstrate contract segmentation on the sample CUAD data."""
    try:
        segmenter = CUADSegmenter()
        segmenter.print_segmentation_summary()
    except Exception as e:
        print(f"Error during segmentation: {e}")
        print("\nMake sure you have:")
        print("1. Set your OPENROUTER_API_KEY in the .env file")
        print("2. The sample_dataset/sample_cuad.json file exists")


if __name__ == "__main__":
    demo_segmentation()