#!/usr/bin/env python3
"""
Process all enterprise PDFs to generate segmentation files.
Required before creating metadata index.
"""

import json
import sys
import os
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import fitz  # PyMuPDF

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from segmentation import segment_document, save_segmentation_to_json
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


class EnterprisePDFProcessor:
    """Process enterprise PDFs for segmentation."""
    
    def __init__(self, enterprise_data_path: str = "data/enterprise/"):
        """Initialize processor."""
        self.enterprise_path = Path(enterprise_data_path)
        self.pdfs_path = self.enterprise_path / "pdfs"
        self.dataset_csv = self.enterprise_path / "dataset.csv"
        self.segmentation_output = Path("output/enterprise_segmentation")
        self.segmentation_output.mkdir(exist_ok=True)
        
        # Load dataset mapping
        self.pdf_metadata = self._load_dataset_csv()
        
    def _load_dataset_csv(self) -> Dict[str, Dict]:
        """Load the dataset CSV mapping SHA1 to company info."""
        metadata = {}
        with open(self.dataset_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row['sha1']] = {
                    'company_name': row['name'],
                    'date': row['date'],
                    'size': int(row['size'])
                }
        return metadata
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_single_pdf(self, pdf_sha1: str) -> bool:
        """Process a single PDF for segmentation."""
        pdf_info = self.pdf_metadata.get(pdf_sha1)
        if not pdf_info:
            print(f"No metadata found for {pdf_sha1}")
            return False
        
        company_name = pdf_info['company_name']
        date = pdf_info['date']
        
        # Check if already processed
        segmentation_file = self.segmentation_output / f"{pdf_sha1}_segmentation.json"
        if segmentation_file.exists():
            print(f"✓ Already processed: {company_name} ({date})")
            return True
        
        print(f"Processing: {company_name} ({date})")
        
        # Extract text from PDF
        pdf_path = self.pdfs_path / f"{pdf_sha1}.pdf"
        if not pdf_path.exists():
            print(f"  ERROR: PDF file not found: {pdf_path}")
            return False
        
        print(f"  Extracting text from PDF...")
        document_text = self.extract_text_from_pdf(pdf_path)
        
        if not document_text.strip():
            print(f"  ERROR: No text extracted from PDF")
            return False
        
        print(f"  Extracted {len(document_text):,} characters")
        
        # Segment document
        print(f"  Segmenting document...")
        try:
            sections = segment_document(document_text, save_results=False)
            print(f"  Created {len(sections)} sections")
            
            # Save segmentation
            segmentation_data = {
                "pdf_sha1": pdf_sha1,
                "company_name": company_name,
                "date": date,
                "timestamp": datetime.now().isoformat(),
                "total_sections": len(sections),
                "total_characters": len(document_text),
                "sections": sections
            }
            
            with open(segmentation_file, 'w', encoding='utf-8') as f:
                json.dump(segmentation_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Saved segmentation to: {segmentation_file}")
            return True
            
        except Exception as e:
            print(f"  ERROR: Segmentation failed: {e}")
            return False
    
    def process_all_pdfs(self) -> Dict[str, Any]:
        """Process all enterprise PDFs."""
        print("Processing all enterprise PDFs for segmentation...")
        print(f"Found {len(self.pdf_metadata)} PDFs to process")
        print(f"Output directory: {self.segmentation_output}")
        print("=" * 60)
        
        results = {
            'start_time': datetime.now().isoformat(),
            'total_pdfs': len(self.pdf_metadata),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'details': []
        }
        
        for i, pdf_sha1 in enumerate(self.pdf_metadata.keys(), 1):
            print(f"\n[{i}/{len(self.pdf_metadata)}] ", end="")
            
            success = self.process_single_pdf(pdf_sha1)
            
            if success:
                results['processed'] += 1
                status = 'success'
            else:
                results['failed'] += 1
                status = 'failed'
            
            results['details'].append({
                'pdf_sha1': pdf_sha1,
                'company': self.pdf_metadata[pdf_sha1]['company_name'],
                'status': status
            })
        
        results['end_time'] = datetime.now().isoformat()
        
        # Save processing results
        results_file = self.segmentation_output / "processing_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n" + "=" * 60)
        print(f"PROCESSING COMPLETE")
        print(f"=" * 60)
        print(f"Total PDFs: {results['total_pdfs']}")
        print(f"Successfully processed: {results['processed']}")
        print(f"Failed: {results['failed']}")
        print(f"Results saved to: {results_file}")
        
        return results


def main():
    """Main function to process enterprise PDFs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process enterprise PDFs for segmentation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Process all PDFs")
    group.add_argument("--single", type=str, help="Process single PDF by SHA1 hash")
    group.add_argument("--list", action="store_true", help="List available PDFs")
    
    args = parser.parse_args()
    
    processor = EnterprisePDFProcessor()
    
    if args.list:
        print("Available PDFs:")
        print("-" * 80)
        for i, (sha1, info) in enumerate(processor.pdf_metadata.items(), 1):
            print(f"{i:2d}. {sha1[:12]}... - {info['company_name']} ({info['date']})")
        return 0
    
    elif args.single:
        sha1 = args.single
        if sha1 not in processor.pdf_metadata:
            print(f"ERROR: SHA1 '{sha1}' not found in dataset")
            print("Use --list to see available PDFs")
            return 1
        
        print(f"Processing single PDF: {sha1}")
        success = processor.process_single_pdf(sha1)
        return 0 if success else 1
    
    elif args.all:
        results = processor.process_all_pdfs()
        
        if results['failed'] > 0:
            print(f"\nFailed PDFs:")
            for detail in results['details']:
                if detail['status'] == 'failed':
                    print(f"  - {detail['company']} ({detail['pdf_sha1']})")
        
        return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)