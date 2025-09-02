#!/usr/bin/env python3
"""
Create metadata index for enterprise financial documents.
Extracts key metadata from PDF segmentation files using LLM to enable question-driven document discovery.
"""

import json
import sys
import os
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import openai
from datetime import datetime
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field, validator

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class CompanyInfo(BaseModel):
    """Company information extracted from financial documents."""
    primary_name: str = Field(description="Official company name")
    aliases: List[str] = Field(default_factory=list, description="Alternative names, ticker symbols")
    industry_sector: str = Field(default="Unknown", description="Industry classification")


class DocumentInfo(BaseModel):
    """Document metadata information."""
    document_type: str = Field(description="Type of document (10-K, 10-Q, 8-K, Annual Report, etc.)")
    fiscal_year: Optional[str] = Field(description="Fiscal year as YYYY string")
    period_end_date: Optional[str] = Field(description="Period end date as YYYY-MM-DD")
    filing_date: Optional[str] = Field(description="Filing date as YYYY-MM-DD")

    @validator('fiscal_year')
    def validate_fiscal_year(cls, v):
        """Ensure fiscal year is valid or None."""
        if v and v.lower() in ['unknown', 'unkn', 'n/a']:
            return None
        return v


class KeySections(BaseModel):
    """Key sections within the financial document."""
    financial_statements: Optional[str] = Field(description="Location of financial statements")
    income_statement: Optional[str] = Field(description="Location of income statement")
    balance_sheet: Optional[str] = Field(description="Location of balance sheet")
    cash_flow: Optional[str] = Field(description="Location of cash flow statement")
    notes: Optional[str] = Field(description="Location of notes to financial statements")


class DocumentMetadata(BaseModel):
    """Complete metadata for a financial document."""
    company_info: CompanyInfo
    document_info: DocumentInfo
    financial_metrics_available: List[str] = Field(default_factory=list, 
                                                  description="Financial metrics available in document")
    key_sections: KeySections
    searchable_keywords: List[str] = Field(default_factory=list,
                                         description="Keywords for document discovery")

try:
    from segmentation import segment_document, save_segmentation_to_json
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


class MetadataExtractor:
    """Extract metadata from enterprise financial documents."""
    
    def __init__(self, enterprise_data_path: str = "data/enterprise/"):
        """Initialize metadata extractor."""
        self.enterprise_path = Path(enterprise_data_path)
        self.segmentation_index = Path("segmentation_results/enterprise/enterprise_segmentation_index.json")
        self.segmentation_results_path = Path("segmentation_results/enterprise")
        
        # Load segmentation index
        self.segmentation_data = self._load_segmentation_index()
        
    def _load_segmentation_index(self) -> Dict:
        """Load the existing enterprise segmentation index."""
        with open(self.segmentation_index, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_segmentation_summary(self, markdown_content: str) -> Dict:
        """
        Extract summary from markdown segmentation file for LLM processing.
        """
        lines = markdown_content.split('\n')
        
        summary = {
            'total_sections': 0,
            'section_headers': [],
            'key_terms_found': [],
            'document_structure': []
        }
        
        current_section = None
        section_num = 0
        
        for line in lines:
            # Look for markdown headers (# ## ### etc.)
            if line.startswith('#'):
                if current_section:
                    # Save previous section
                    summary['document_structure'].append(current_section)
                
                section_num += 1
                header = line.lstrip('#').strip()[:100]  # Remove # and limit length
                summary['section_headers'].append(header)
                
                current_section = {
                    'section_num': section_num,
                    'title': header,
                    'content_preview': '',
                    'char_count': 0
                }
            elif current_section and line.strip():
                # Collect content preview
                if len(current_section['content_preview']) < 200:
                    current_section['content_preview'] += line.strip() + ' '
                current_section['char_count'] += len(line)
        
        # Don't forget the last section
        if current_section:
            summary['document_structure'].append(current_section)
        
        summary['total_sections'] = len(summary['document_structure'])
        
        # Look for financial keywords
        financial_keywords = [
            'revenue', 'income', 'profit', 'loss', 'assets', 'liabilities', 
            'cash flow', 'earnings', 'ebitda', 'margin', 'equity',
            'balance sheet', 'income statement', 'financial position'
        ]
        
        full_text = markdown_content.lower()
        for keyword in financial_keywords:
            if keyword in full_text:
                summary['key_terms_found'].append(keyword)
        
        return summary
    
    def extract_segmentation_summary_from_json(self, segmentation_data: Dict) -> Dict:
        """
        Extract summary from JSON segmentation data for LLM processing.
        Uses summary fields instead of content previews for efficiency.
        """
        sections = segmentation_data.get('sections', [])
        
        summary = {
            'total_sections': len(sections),
            'section_headers': [],
            'key_terms_found': [],
            'document_structure': []
        }
        
        all_key_terms = set()
        
        for i, section in enumerate(sections):
            # Extract headers and summaries
            header = section.get('title', f'Section {i+1}')[:100]  # Limit length
            section_summary = section.get('summary', '')[:200]  # Use summary instead of content preview
            financial_concept = section.get('financial_concept', '')
            
            summary['section_headers'].append(header)
            summary['document_structure'].append({
                'section_num': i + 1,
                'title': header,
                'summary': section_summary,
                'financial_concept': financial_concept,
                'char_count': len(section.get('text', ''))
            })
            
            # Collect key terms from each section
            section_key_terms = section.get('key_terms', [])
            all_key_terms.update([term.lower() for term in section_key_terms])
        
        # Financial keywords from the structured data
        financial_keywords = [
            'revenue', 'income', 'profit', 'loss', 'assets', 'liabilities', 
            'cash flow', 'earnings', 'ebitda', 'margin', 'equity',
            'balance sheet', 'income statement', 'financial position'
        ]
        
        # Find which financial keywords are present in key terms
        for keyword in financial_keywords:
            if any(keyword in term for term in all_key_terms):
                summary['key_terms_found'].append(keyword)
        
        return summary
    
    def generate_metadata_with_llm(self, company_name: str, date: str, 
                                  segmentation_summary: Dict) -> DocumentMetadata:
        """
        Use Instructor with LLM to extract structured metadata from segmentation summary.
        """
        try:
            api_key = os.getenv('OPENROUTER_API_KEY')
            model_name = os.getenv('MODEL_NAME', 'openai/gpt-4o-mini')
            
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            
            # Create instructor client
            client = instructor.from_openai(
                openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
            )
            
            # Create structured prompt
            prompt = f"""
Analyze this financial document segmentation summary and extract key metadata for document indexing.

Company: {company_name}
Date: {date}

Document Structure Summary:
- Total sections: {segmentation_summary['total_sections']}
- Section headers: {segmentation_summary['section_headers'][:10]}
- Key financial terms found: {segmentation_summary['key_terms_found']}

Sample section details:
{json.dumps(segmentation_summary['document_structure'][:5], indent=2)}

Extract metadata to help match questions to this specific document. Focus on factual information from the document structure.
"""
            
            # Use instructor to get structured response
            metadata = client.chat.completions.create(
                model=model_name,
                response_model=DocumentMetadata,
                messages=[
                    {"role": "system", "content": "You are a financial document analyst. Extract metadata accurately based on the document structure provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return metadata
            
        except Exception as e:
            print(f"LLM extraction failed for {company_name}: {e}")
            # Return basic fallback metadata using Pydantic models
            return DocumentMetadata(
                company_info=CompanyInfo(
                    primary_name=company_name,
                    aliases=[],
                    industry_sector="Unknown"
                ),
                document_info=DocumentInfo(
                    document_type="Unknown",
                    fiscal_year=date[:4] if len(date) >= 4 and date != "unknown" else None,
                    period_end_date=date if date != "unknown" else None,
                    filing_date=None
                ),
                financial_metrics_available=[],
                key_sections=KeySections(),
                searchable_keywords=[company_name.lower()]
            )
    
    def process_document_for_metadata(self, doc_info: Dict) -> Dict:
        """Process a single document to extract metadata."""
        company_name = doc_info['company_name']
        filename = doc_info['filename']
        doc_type = doc_info['document_type']
        
        print(f"Processing {company_name} ({doc_type})")
        
        # Find corresponding JSON segmentation file
        # Convert from "Holley_Inc_194000c9109c6fa628f1fed33b44ae4c2b8365f4.md" 
        # to "Holley_Inc_10-K_segmented_*.json"
        base_name = filename.replace('.md', '').rsplit('_', 1)[0]  # Remove SHA1 and .md
        json_pattern = f"{base_name}_{doc_type}_segmented_*.json"
        
        # Find matching JSON files
        json_files = list(self.segmentation_results_path.glob(json_pattern))
        if not json_files:
            print(f"  ERROR: No JSON segmentation file found matching: {json_pattern}")
            return None
        
        # Use the most recent one (they're timestamped)
        json_file = sorted(json_files)[-1]
        print(f"  Loading segmentation from: {json_file.name}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            segmentation_data = json.load(f)
        
        # Extract summary for LLM
        summary = self.extract_segmentation_summary_from_json(segmentation_data)
        
        # Generate metadata using LLM (use unknown date since not in segmentation index)
        metadata_pydantic = self.generate_metadata_with_llm(company_name, "unknown", summary)
        
        # Extract SHA1 from filename
        sha1 = filename.split('_')[-1].replace('.md', '')
        
        # Convert Pydantic model to dict and add file reference
        metadata = metadata_pydantic.dict()
        metadata['file_info'] = {
            'sha1': sha1,
            'company_name': company_name,
            'segmentation_path': str(json_file),
            'document_type': doc_type,
            'total_sections': doc_info['total_sections']
        }
        
        return metadata
    
    def create_metadata_index(self) -> Dict:
        """Create complete metadata index for all enterprise documents."""
        print("Creating metadata index for enterprise documents...")
        
        documents = self.segmentation_data['documents']
        
        metadata_index = {
            'creation_timestamp': datetime.now().isoformat(),
            'total_documents': len(documents),
            'companies': {},
            'by_year': {},
            'by_document_type': {},
            'searchable_index': {}
        }
        
        for doc_info in documents:
            metadata = self.process_document_for_metadata(doc_info)
            
            if metadata is None:
                print(f"  Skipping {doc_info['company_name']} - processing failed")
                continue
            
            company_name = metadata['company_info']['primary_name']
            fiscal_year = metadata['document_info']['fiscal_year']
            doc_type = metadata['document_info']['document_type']
            sha1 = metadata['file_info']['sha1']
            
            # Handle None/Unknown fiscal years
            if fiscal_year in [None, 'Unknown', 'unkn']:
                fiscal_year = 'Unknown'
            
            # Index by company
            if company_name not in metadata_index['companies']:
                metadata_index['companies'][company_name] = {}
            metadata_index['companies'][company_name][fiscal_year] = metadata
            
            # Index by year
            if fiscal_year not in metadata_index['by_year']:
                metadata_index['by_year'][fiscal_year] = []
            metadata_index['by_year'][fiscal_year].append(metadata)
            
            # Index by document type
            if doc_type not in metadata_index['by_document_type']:
                metadata_index['by_document_type'][doc_type] = []
            metadata_index['by_document_type'][doc_type].append(metadata)
            
            # Build searchable keywords index
            all_keywords = [company_name.lower()]
            all_keywords.extend([alias.lower() for alias in metadata['company_info']['aliases']])
            all_keywords.extend([kw.lower() for kw in metadata['searchable_keywords']])
            
            for keyword in all_keywords:
                if keyword not in metadata_index['searchable_index']:
                    metadata_index['searchable_index'][keyword] = []
                metadata_index['searchable_index'][keyword].append({
                    'sha1': sha1,
                    'company': company_name,
                    'year': fiscal_year,
                    'metrics': metadata['financial_metrics_available']
                })
        
        # Save metadata index in same folder as segmentation
        index_path = Path("segmentation_results/enterprise/metadata_index.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_index, f, indent=2, ensure_ascii=False)
        
        print(f"\nMetadata index saved to: {index_path}")
        print(f"Indexed {len(metadata_index['companies'])} companies")
        
        return metadata_index


def main():
    """Main function to create metadata index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create metadata index for enterprise documents")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--single", type=str, help="Process single company by name")
    group.add_argument("--all", action="store_true", help="Process all documents", default=True)
    
    args = parser.parse_args()
    
    extractor = MetadataExtractor()
    
    if args.single:
        # Find and process single document
        documents = extractor.segmentation_data['documents']
        matching_docs = [doc for doc in documents if args.single.lower() in doc['company_name'].lower()]
        
        if not matching_docs:
            print(f"No documents found matching: {args.single}")
            print("Available companies:")
            for doc in documents:
                print(f"  - {doc['company_name']}")
            return 1
        
        doc_info = matching_docs[0]
        print(f"Processing single document: {doc_info['company_name']}")
        
        metadata = extractor.process_document_for_metadata(doc_info)
        
        if metadata:
            print("\n" + "="*60)
            print("SINGLE DOCUMENT METADATA")
            print("="*60)
            print(json.dumps(metadata, indent=2))
        else:
            print("Failed to process document")
            return 1
    
    else:
        # Create the full metadata index
        metadata_index = extractor.create_metadata_index()
        
        # Print summary
        print("\n" + "="*60)
        print("METADATA INDEX SUMMARY")
        print("="*60)
        print(f"Companies indexed: {len(metadata_index['companies'])}")
        # Filter out None values and sort years
        valid_years = [year for year in metadata_index['by_year'].keys() if year is not None and year != 'Unknown']
        print(f"Years covered: {sorted(valid_years) if valid_years else 'No valid years found'}")
        print(f"Document types: {list(metadata_index['by_document_type'].keys())}")
        print(f"Searchable keywords: {len(metadata_index['searchable_index'])}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)