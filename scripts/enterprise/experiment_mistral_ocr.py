#!/usr/bin/env python3
"""
Experimental script to test Mistral OCR with a single PDF.
This will help us understand the output format before building the full processor.
"""

import os
import base64
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral

def load_sample_pdf():
    """Load the first PDF from our dataset for testing."""
    project_root = Path(__file__).parent.parent.parent
    pdfs_dir = project_root / "data" / "enterprise" / "pdfs"
    
    # Get first PDF file
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in data/enterprise/pdfs/")
    
    sample_pdf = pdf_files[0]  # Take first PDF
    print(f"Using sample PDF: {sample_pdf.name}")
    print(f"File size: {sample_pdf.stat().st_size / 1024 / 1024:.2f} MB")
    
    return sample_pdf

def pdf_to_base64(pdf_path):
    """Convert PDF to base64 for API processing."""
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    return base64.b64encode(pdf_bytes).decode('utf-8')

def test_mistral_ocr(pdf_path):
    """Test Mistral OCR with a single PDF and examine the output."""
    load_dotenv()
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found. Please add it to your .env file")
    
    client = Mistral(api_key=api_key)
    
    print("Converting PDF to base64...")
    start_time = time.time()
    pdf_base64 = pdf_to_base64(pdf_path)
    print(f"Base64 conversion took: {time.time() - start_time:.2f}s")
    print(f"Base64 length: {len(pdf_base64)} characters")
    
    print("\nSending to Mistral OCR...")
    ocr_start = time.time()
    
    try:
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_base64}"
            }
        )
        
        ocr_time = time.time() - ocr_start
        print(f"OCR processing took: {ocr_time:.2f}s")
        
        # Examine the response structure
        print("\n" + "="*60)
        print("RESPONSE STRUCTURE ANALYSIS")
        print("="*60)
        
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {dir(response)}")
        
        # Try to access different possible attributes
        possible_attrs = ['text', 'content', 'pages', 'data', 'result', 'output']
        
        for attr in possible_attrs:
            if hasattr(response, attr):
                value = getattr(response, attr)
                print(f"\n{attr}: {type(value)}")
                if isinstance(value, (str, int, float, bool)):
                    preview = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                    print(f"  Content preview: {preview}")
                elif isinstance(value, (list, dict)):
                    print(f"  Length/Keys: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                    if isinstance(value, dict) and len(value) < 10:
                        print(f"  Keys: {list(value.keys())}")
                    elif isinstance(value, list) and len(value) > 0:
                        print(f"  First item type: {type(value[0])}")
        
        # Try to convert response to dict/JSON
        try:
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
            elif hasattr(response, 'dict'):
                response_dict = response.dict()
            else:
                response_dict = vars(response) if hasattr(response, '__dict__') else str(response)
            
            print(f"\nResponse as dict keys: {list(response_dict.keys()) if isinstance(response_dict, dict) else 'Not a dict'}")
            
        except Exception as e:
            print(f"Could not convert to dict: {e}")
        
        # Save raw response for inspection
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "output" / "enterprise"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save response structure
        response_analysis = {
            "pdf_file": pdf_path.name,
            "processing_time": ocr_time,
            "response_type": str(type(response)),
            "response_attributes": dir(response),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Try to extract and save actual content
        content_found = None
        for attr in ['text', 'content', 'pages', 'data']:
            if hasattr(response, attr):
                attr_value = getattr(response, attr)
                response_analysis[f"{attr}_type"] = str(type(attr_value))
                response_analysis[f"{attr}_preview"] = str(attr_value)[:500] if attr_value else None
                
                if attr_value and not content_found:
                    content_found = attr_value
        
        # Save analysis
        with open(output_dir / "mistral_ocr_experiment.json", 'w', encoding='utf-8') as f:
            json.dump(response_analysis, f, indent=2, ensure_ascii=False)
        
        # Save actual content if found
        if content_found:
            with open(output_dir / "sample_ocr_output.txt", 'w', encoding='utf-8') as f:
                f.write(str(content_found))
            print(f"\nContent saved to: {output_dir / 'sample_ocr_output.txt'}")
        
        print(f"\nExperiment results saved to: {output_dir / 'mistral_ocr_experiment.json'}")
        
        return response
        
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return None

def main():
    """Main experimental function."""
    print("Mistral OCR Experiment")
    print("=" * 40)
    
    try:
        # Load sample PDF
        sample_pdf = load_sample_pdf()
        
        # Test OCR
        response = test_mistral_ocr(sample_pdf)
        
        if response:
            print("\n✅ OCR experiment completed successfully!")
            print("Check the output files to understand the response format.")
        else:
            print("\n❌ OCR experiment failed.")
            
    except Exception as e:
        print(f"Experiment failed: {e}")

if __name__ == "__main__":
    main()