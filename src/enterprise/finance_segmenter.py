"""
Enterprise financial document segmentation specialized for SEC filings.
Handles 10-K, 10-Q, and annual reports from the enterprise dataset.
"""

import json
import os
from typing import List, Dict, Any, Optional
from .finance_segmentation import segment_financial_document, save_financial_segmentation_to_json


class EnterpriseFinanceSegmenter:
    """Specialized segmenter for enterprise financial documents."""
    
    def __init__(self, enterprise_data_path: str = "data/enterprise/markdown"):
        """
        Initialize with enterprise data path.
        
        Args:
            enterprise_data_path: Path to the enterprise markdown files
        """
        self.enterprise_data_path = enterprise_data_path
        self.document_files = self._discover_documents()
    
    def _discover_documents(self) -> List[Dict[str, str]]:
        """Discover all markdown documents in the enterprise data path."""
        documents = []
        
        if not os.path.exists(self.enterprise_data_path):
            raise FileNotFoundError(f"Enterprise data path not found: {self.enterprise_data_path}")
        
        for filename in os.listdir(self.enterprise_data_path):
            if filename.endswith('.md'):
                # Extract company name from filename
                # Format: CompanyName_hash.md
                company_name = filename.replace('.md', '').split('_')[:-1]  # Remove hash part
                company_name = '_'.join(company_name).replace('_', ' ')
                
                documents.append({
                    'filename': filename,
                    'filepath': os.path.join(self.enterprise_data_path, filename),
                    'company_name': company_name
                })
        
        print(f"Discovered {len(documents)} enterprise documents")
        return documents
    
    def _load_document_content(self, filepath: str) -> str:
        """Load the content of a markdown document."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"Document not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error reading document {filepath}: {e}")
    
    def _detect_document_type(self, content: str) -> str:
        """
        Detect the type of financial document based on content.
        
        Args:
            content: Document content
            
        Returns:
            Document type ('10-K', '10-Q', '20-F', 'annual-report')
        """
        content_upper = content.upper()
        
        if 'FORM 10-K' in content_upper:
            return '10-K'
        elif 'FORM 10-Q' in content_upper:
            return '10-Q'
        elif 'FORM 20-F' in content_upper:
            return '20-F'
        elif 'ANNUAL REPORT' in content_upper:
            return 'annual-report'
        else:
            # Default to 10-K if we can't determine
            return '10-K'
    
    def _get_cache_filepath(self, company_name: str, document_type: str) -> str:
        """Get the filepath for cached segmentation results."""
        safe_company_name = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_company_name = safe_company_name.replace(' ', '_')[:50]  # Limit length
        return f"segmentation_results/enterprise/{safe_company_name}_{document_type}_cached.json"
    
    def segment_document(self, filename: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Segment a specific enterprise document.
        
        Args:
            filename: Name of the document file
            use_cache: Whether to use cached results if available
        
        Returns:
            Dictionary containing segmentation results
        """
        # Find the document
        doc_info = None
        for doc in self.document_files:
            if doc['filename'] == filename:
                doc_info = doc
                break
        
        if not doc_info:
            raise ValueError(f"Document {filename} not found in enterprise data")
        
        company_name = doc_info['company_name']
        filepath = doc_info['filepath']
        
        # Load document content
        content = self._load_document_content(filepath)
        
        # Detect document type
        document_type = self._detect_document_type(content)
        
        cache_filepath = self._get_cache_filepath(company_name, document_type)
        
        # Try to load from cache first
        if use_cache and os.path.exists(cache_filepath):
            try:
                print(f"Loading cached segmentation from: {cache_filepath}")
                with open(cache_filepath, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                return {
                    "company_name": cached_data["company_name"],
                    "document_type": cached_data["document_type"],
                    "sections": cached_data["sections"],
                    "total_sections": cached_data["total_sections"],
                    "cached": True
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Proceeding with new segmentation...")
        
        # Perform new segmentation
        print(f"Performing new segmentation for {company_name} ({document_type})...")
        
        # Segment the document
        sections = segment_financial_document(
            content, 
            company_name, 
            document_type, 
            save_results=False  # We'll save manually with cache
        )
        
        # Save results and cache
        try:
            # Save main result
            save_financial_segmentation_to_json(sections, company_name, document_type)
            
            # Save cache file
            os.makedirs("segmentation_results/enterprise", exist_ok=True)
            result_data = {
                "company_name": company_name,
                "document_type": document_type,
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
            "company_name": company_name,
            "document_type": document_type,
            "sections": sections,
            "total_sections": len(sections),
            "cached": False
        }
    
    def segment_all_documents(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Segment all enterprise documents.
        
        Args:
            use_cache: Whether to use cached results if available
            
        Returns:
            Dictionary with results for all documents
        """
        return self.segment_selected_documents(self.document_files, use_cache)
    
    def segment_selected_documents(self, docs_to_process: List[Dict], use_cache: bool = True) -> Dict[str, Any]:
        """
        Segment selected enterprise documents.
        
        Args:
            docs_to_process: List of document info dictionaries to process
            use_cache: Whether to use cached results if available
            
        Returns:
            Dictionary with results for processed documents
        """
        results = {
            "timestamp": json.dumps(None),
            "total_documents": len(docs_to_process),
            "processed_documents": 0,
            "failed_documents": 0,
            "documents": []
        }
        
        for doc_info in docs_to_process:
            try:
                print(f"\n{'='*80}")
                print(f"Processing: {doc_info['company_name']}")
                print(f"File: {doc_info['filename']}")
                print(f"{'='*80}")
                
                result = self.segment_document(doc_info['filename'], use_cache)
                
                results["documents"].append({
                    "filename": doc_info['filename'],
                    "company_name": result["company_name"],
                    "document_type": result["document_type"],
                    "total_sections": result["total_sections"],
                    "cached": result.get("cached", False),
                    "status": "success"
                })
                
                results["processed_documents"] += 1
                print(f"✅ Successfully processed {doc_info['company_name']} - {result['total_sections']} sections")
                
            except Exception as e:
                print(f"❌ Failed to process {doc_info['filename']}: {e}")
                results["documents"].append({
                    "filename": doc_info['filename'],
                    "company_name": doc_info['company_name'],
                    "error": str(e),
                    "status": "failed"
                })
                results["failed_documents"] += 1
        
        # Save master index
        self._save_master_index(results)
        
        return results
    
    def _save_master_index(self, results: Dict[str, Any]) -> str:
        """Save master index of all processed documents."""
        from datetime import datetime
        
        results["timestamp"] = datetime.now().isoformat()
        
        os.makedirs("segmentation_results/enterprise", exist_ok=True)
        index_filepath = "segmentation_results/enterprise/enterprise_segmentation_index.json"
        
        with open(index_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Master index saved to: {index_filepath}")
        return index_filepath
    
    def print_segmentation_summary(self, filename: str) -> None:
        """Print a detailed summary of document segmentation."""
        result = self.segment_document(filename)
        
        cache_status = " (from cache)" if result.get('cached', False) else " (newly generated)"
        print(f"\n{'='*120}")
        print(f"Company: {result['company_name']}")
        print(f"Document Type: {result['document_type']}")
        print(f"Total Sections: {result['total_sections']}{cache_status}")
        print(f"{'='*120}")
        
        for i, section in enumerate(result['sections']):
            print(f"\nSection {i+1}: {section['title']}")
            
            if section.get('item_number'):
                print(f"SEC Item: {section['item_number']}")
            
            print(f"Financial Concept: {section.get('financial_concept', 'N/A')}")
            print(f"Lines: {section['start_index']}-{section['end_index']} ({section['line_count']} lines)")
            
            # Display summary
            summary = section.get('summary', 'No summary available')
            try:
                print(f"Summary: {summary}")
            except UnicodeEncodeError:
                summary_clean = summary.encode('ascii', 'replace').decode('ascii')
                print(f"Summary: {summary_clean}")
            
            # Display key terms
            if section.get('key_terms'):
                key_terms_str = ', '.join(section['key_terms'])
                try:
                    print(f"Key Terms: {key_terms_str}")
                except UnicodeEncodeError:
                    key_terms_clean = key_terms_str.encode('ascii', 'replace').decode('ascii')
                    print(f"Key Terms: {key_terms_clean}")
            
            # Display financial metrics
            if section.get('financial_metrics'):
                metrics_str = ', '.join(section['financial_metrics'])
                try:
                    print(f"Financial Metrics: {metrics_str}")
                except UnicodeEncodeError:
                    metrics_clean = metrics_str.encode('ascii', 'replace').decode('ascii')
                    print(f"Financial Metrics: {metrics_clean}")
            
            # Display reasoning
            if section.get('reasoning'):
                try:
                    print(f"Reasoning: {section['reasoning']}")
                except UnicodeEncodeError:
                    reasoning_clean = section['reasoning'].encode('ascii', 'replace').decode('ascii')
                    print(f"Reasoning: {reasoning_clean}")
            
            # Show content preview
            section_text = section['text'].strip()
            lines = section_text.split('\\n')[:3]
            preview = '\\n'.join(lines)
            
            print(f"Content Preview:")
            try:
                print(f"   {preview}")
            except UnicodeEncodeError:
                preview_clean = preview.encode('ascii', 'replace').decode('ascii')
                print(f"   {preview_clean}")
            
            if len(section['text'].split('\\n')) > 3:
                print(f"   ... ({len(section['text'].split('\\n')) - 3} more lines)")
            
            print(f"{'-'*120}")
    
    def list_documents(self) -> None:
        """List all available enterprise documents."""
        print(f"\n📁 Enterprise Documents ({len(self.document_files)} found):")
        print(f"{'='*80}")
        
        for i, doc in enumerate(self.document_files, 1):
            print(f"{i:2d}. {doc['company_name']}")
            print(f"    File: {doc['filename']}")
            
            # Check if already processed
            content = self._load_document_content(doc['filepath'])
            doc_type = self._detect_document_type(content)
            cache_path = self._get_cache_filepath(doc['company_name'], doc_type)
            
            if os.path.exists(cache_path):
                print(f"    Status: ✅ Already segmented ({doc_type})")
            else:
                print(f"    Status: ⏳ Not yet processed ({doc_type})")
            print()


def demo_enterprise_segmentation():
    """Demonstrate enterprise document segmentation."""
    try:
        segmenter = EnterpriseFinanceSegmenter()
        
        print("Available enterprise documents:")
        segmenter.list_documents()
        
        if segmenter.document_files:
            # Demo with first document
            first_doc = segmenter.document_files[0]['filename']
            print(f"\nDemo segmentation with: {first_doc}")
            segmenter.print_segmentation_summary(first_doc)
        else:
            print("No enterprise documents found!")
            
    except Exception as e:
        print(f"Error during enterprise segmentation demo: {e}")
        print("\nMake sure you have:")
        print("1. Set your OPENROUTER_API_KEY in the .env file")
        print("2. Enterprise markdown files in data/enterprise/markdown/")


if __name__ == "__main__":
    demo_enterprise_segmentation()