#!/usr/bin/env python3
"""
Batch processing script to segment all enterprise financial documents.
Processes 10-K, 10-Q, and annual reports into structured JSON segments.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from enterprise.finance_segmenter import EnterpriseFinanceSegmenter


def main():
    """Main function to process all enterprise documents."""
    print("🏢 Enterprise Financial Document Segmentation")
    print("=" * 80)
    
    try:
        # Initialize the segmenter
        print("Initializing Enterprise Finance Segmenter...")
        segmenter = EnterpriseFinanceSegmenter()
        
        # List all available documents
        print(f"\n📋 Found {len(segmenter.document_files)} enterprise documents")
        segmenter.list_documents()
        
        # Ask user if they want to proceed
        print(f"\n🚀 Processing options:")
        print("1. Process all documents")
        print("2. Process first 2 documents (test mode)")
        print("3. Cancel")
        
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == '1':
            docs_to_process = segmenter.document_files
            print(f"Processing all {len(docs_to_process)} documents...")
        elif choice == '2':
            docs_to_process = segmenter.document_files[:2]
            print(f"Processing first 2 documents for testing...")
        else:
            print("Operation cancelled.")
            return
        
        print(f"\n🔄 Starting batch processing...")
        print("=" * 80)
        
        # Process selected documents
        results = segmenter.segment_selected_documents(docs_to_process, use_cache=True)
        
        # Print summary
        print(f"\n📊 PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Total Documents: {results['total_documents']}")
        print(f"Successfully Processed: {results['processed_documents']}")
        print(f"Failed: {results['failed_documents']}")
        print(f"Success Rate: {(results['processed_documents'] / results['total_documents'] * 100):.1f}%")
        
        # Print details for each document
        print(f"\n📋 DETAILED RESULTS")
        print("=" * 80)
        
        for doc in results['documents']:
            status_icon = "✅" if doc['status'] == 'success' else "❌"
            cache_info = " (cached)" if doc.get('cached') else " (new)"
            
            if doc['status'] == 'success':
                print(f"{status_icon} {doc['company_name']}")
                print(f"   Type: {doc['document_type']} | Sections: {doc['total_sections']}{cache_info}")
            else:
                print(f"{status_icon} {doc['company_name']} - FAILED")
                print(f"   Error: {doc.get('error', 'Unknown error')}")
            print()
        
        # Show output location
        output_dir = project_root / "segmentation_results" / "enterprise"
        print(f"📁 Results saved to: {output_dir}")
        print(f"📋 Master index: {output_dir / 'enterprise_segmentation_index.json'}")
        
        if results['processed_documents'] > 0:
            print(f"\n🎉 Successfully processed {results['processed_documents']} financial documents!")
            print("You can now use these segmented documents for RAG applications.")
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user.")
        return
    except Exception as e:
        print(f"\n❌ Error during batch processing: {e}")
        print("\nMake sure you have:")
        print("1. Set your OPENROUTER_API_KEY in the .env file")
        print("2. Enterprise markdown files in data/enterprise/markdown/")
        sys.exit(1)


def demo_single_document():
    """Demo function to process a single document."""
    print("🔍 Enterprise Document Segmentation Demo")
    print("=" * 80)
    
    try:
        segmenter = EnterpriseFinanceSegmenter()
        
        if not segmenter.document_files:
            print("❌ No enterprise documents found!")
            return
        
        # Show available documents
        print("Available documents:")
        for i, doc in enumerate(segmenter.document_files, 1):
            print(f"{i:2d}. {doc['company_name']} ({doc['filename']})")
        
        # Get user selection
        try:
            choice = int(input(f"\nSelect document (1-{len(segmenter.document_files)}): ")) - 1
            if 0 <= choice < len(segmenter.document_files):
                selected_doc = segmenter.document_files[choice]['filename']
                print(f"\n🔄 Processing: {segmenter.document_files[choice]['company_name']}")
                segmenter.print_segmentation_summary(selected_doc)
            else:
                print("❌ Invalid selection.")
        except (ValueError, IndexError):
            print("❌ Invalid input.")
            
    except Exception as e:
        print(f"❌ Error during demo: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_single_document()
    else:
        main()