"""
Financial document segmentation module specialized for SEC filings.
Extends the base segmentation functionality for 10-K, 10-Q, and annual reports.
"""

import os
import time
import json
from typing import List, Optional
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class FinancialSection(BaseModel):
    """Represents a meaningful section of a financial document with SEC filing structure."""
    title: str = Field(description="Main topic of this section of the financial document")
    start_index: int = Field(description="Line number where the section begins")
    end_index: int = Field(description="Line number where the section ends")
    item_number: Optional[str] = Field(description="SEC filing item number if applicable (e.g., 'Item 1', 'Item 1A', 'Item 7')")
    financial_concept: str = Field(description="Primary financial area this section covers (e.g., 'Business Overview', 'Risk Factors', 'Financial Performance')")
    summary: str = Field(description="2-3 sentence summary including key financial information and metrics")
    key_terms: List[str] = Field(description="List of important financial and business terms from this section")
    financial_metrics: List[str] = Field(description="Any financial metrics, ratios, or numerical data mentioned (e.g., 'revenue', 'net income', 'debt-to-equity')")
    reasoning: str = Field(description="Chain of thought reasoning about why this section was identified and its financial significance")


class StructuredFinancialDocument(BaseModel):
    """Financial document with meaningful sections, organized by SEC filing structure."""
    sections: List[FinancialSection] = Field(description="A list of sections of the financial document")


def get_structured_financial_document(document_with_line_numbers: str, 
                                     document_type: str = "10-K",
                                     max_retries: int = 3) -> StructuredFinancialDocument:
    """
    Use LLM to segment financial document into meaningful sections with retry logic.
    
    Args:
        document_with_line_numbers: Document text with line numbers
        document_type: Type of document ('10-K', '10-Q', 'annual-report')
        max_retries: Maximum number of retry attempts
        
    Returns:
        StructuredFinancialDocument with identified sections
    """
    # Import from parent directory
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from segmentation import create_instructor_client
    
    client = create_instructor_client()
    model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
    
    print(f"Using model: {model_name}")
    print(f"Document type: {document_type}")
    print(f"Document length: {len(document_with_line_numbers)} characters")
    
    system_prompt = f"""You are a world class financial document analyzer specializing in SEC filings and corporate financial reports.

Your task is to create a StructuredFinancialDocument with detailed sections optimized for financial analysis and research.

DOCUMENT TYPE: {document_type}

CRITICAL REQUIREMENT: You MUST segment the ENTIRE document starting from line [0]. Include ALL content including document headers, company identification, table of contents, and front matter. The first section MUST start at line [0] or [1]. Do not skip any content at the beginning of the document.

CHAIN OF THOUGHT PROCESS:
For each section, think step by step:

Step 1: IDENTIFY regulatory structure - Where does one SEC Item or business section end and another begin?
Step 2: ANALYZE financial concept - What area of business/finance does this section cover?
Step 3: EXTRACT key financial information - What are the main metrics, performance indicators, risks, or business details?
Step 4: IDENTIFY searchable terms - What keywords would analysts, investors, or researchers search for to find this content?
Step 5: CAPTURE financial metrics - What specific numbers, ratios, percentages, or financial data are mentioned?
Step 6: SYNTHESIZE into summary - Create a 2-3 sentence summary that includes key business and financial insights.

INPUT DOCUMENT FORMAT:
Each line is marked with its line number in square brackets (e.g. [0], [1], [2], [3], etc). Use these line numbers for start_index and end_index.

DOCUMENT SECTIONS to identify (in order from beginning):
- Document Header/Cover Page (company name, filing type, date)
- Table of Contents (index of sections and page numbers)
- Forward-Looking Statements (risk disclaimers)
- SEC Filing Information (commission file number, registration details)

SEC FILING SECTIONS to identify (for 10-K documents):
- Item 1: Business (company operations, products, services, competition)
- Item 1A: Risk Factors (business risks, market risks, operational risks)
- Item 1B: Unresolved Staff Comments
- Item 2: Properties (real estate, facilities)
- Item 3: Legal Proceedings
- Item 7: Management's Discussion and Analysis (MD&A - financial performance analysis)
- Item 7A: Market Risk Disclosures
- Item 8: Financial Statements (balance sheet, income statement, cash flow)
- Item 9A: Controls and Procedures
- Other regulatory sections

BUSINESS/FINANCIAL CONCEPTS to consider:
- Document identification and metadata
- Business strategy and operations
- Financial performance and metrics
- Risk assessment and management
- Market position and competition
- Regulatory compliance
- Corporate governance
- Capital structure and financing
- Revenue streams and cost structure
- Forward-looking statements and guidance

For each section, provide:
- title: Clear, descriptive title of the business/financial concept
- item_number: SEC item number if this is a formal SEC filing section (e.g., "Item 1A", "Item 7")
- financial_concept: The primary business/financial area (e.g., "Business Operations", "Risk Management", "Financial Performance")
- summary: 2-3 sentences describing key business and financial insights with specific metrics when available
- key_terms: Array of important business and financial keywords for searching
- financial_metrics: Array of specific financial data mentioned (revenue figures, ratios, percentages, etc.)
- reasoning: Brief explanation of why this section is financially/strategically significant

EXAMPLE REASONING:
"This section covers the company's risk factors because it outlines specific business, operational, and market risks that could impact financial performance. The detailed discussion of supply chain disruption and regulatory compliance risks are critical for investment analysis and due diligence."

Make sections meaningful, comprehensive, and optimized for financial research and analysis."""
    
    for attempt in range(max_retries):
        try:
            print(f"Sending request to OpenRouter... (Attempt {attempt + 1}/{max_retries})")
            response = client.chat.completions.create(
                model=model_name,
                response_model=StructuredFinancialDocument,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user", 
                        "content": document_with_line_numbers,
                    },
                ],
            )
            print("Response received successfully!")
            return response
            
        except Exception as e:
            print(f"API Error on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("All retry attempts failed")
                raise


def get_financial_sections_text(structured_doc: StructuredFinancialDocument, 
                               line2text: dict[int, str]) -> List[dict]:
    """
    Extract the actual text content for each financial section with enhanced metadata.
    
    Args:
        structured_doc: The structured financial document with sections
        line2text: Mapping from line numbers to text
        
    Returns:
        List of dictionaries containing section info, text, and financial metadata
    """
    sections = []
    
    # Calculate character positions for each section
    char_position = 0
    
    for section in structured_doc.sections:
        # Extract text from start_index to end_index
        section_lines = []
        for line_num in range(section.start_index, min(section.end_index + 1, len(line2text))):
            if line_num in line2text:
                section_lines.append(line2text[line_num])
        
        section_text = "\n".join(section_lines)
        
        # Calculate character start position for this section
        section_char_start = char_position
        section_char_end = char_position + len(section_text)
        
        sections.append({
            "title": section.title,
            "start_index": section.start_index,
            "end_index": section.end_index,
            "char_start": section_char_start,
            "char_end": section_char_end,
            "text": section_text,
            "line_count": section.end_index - section.start_index + 1,
            "item_number": section.item_number,
            "financial_concept": section.financial_concept,
            "summary": section.summary,
            "key_terms": section.key_terms,
            "financial_metrics": section.financial_metrics,
            "reasoning": section.reasoning,
            # Create searchable content combining all relevant fields
            "searchable_content": f"{section.title} | {section.item_number or ''} | {section.financial_concept} | {section.summary} | {' '.join(section.key_terms)} | {' '.join(section.financial_metrics)}"
        })
        
        # Update character position for next section (add newline between sections)
        char_position = section_char_end + 1
    
    return sections


def segment_financial_document(document: str, 
                             company_name: str,
                             document_type: str = "10-K",
                             save_results: bool = True) -> List[dict]:
    """
    Complete pipeline to segment a financial document into meaningful sections.
    Handles large documents by chunking them intelligently.
    
    Args:
        document: The original document text
        company_name: Name of the company (for saving results)
        document_type: Type of document ('10-K', '10-Q', 'annual-report')
        save_results: Whether to save results to JSON file
        
    Returns:
        List of section dictionaries with titles, indices, and text content
    """
    from .document_chunker import DocumentChunker
    
    # Import from parent directory
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from segmentation import doc_with_lines
    
    # Check if document needs chunking
    chunker = DocumentChunker()
    estimated_tokens = chunker.estimate_tokens(document)
    
    if estimated_tokens <= chunker.max_tokens:
        # Process normally for smaller documents
        return _segment_single_document(document, company_name, document_type, save_results)
    else:
        # Process large document with chunking
        return _segment_chunked_document(document, company_name, document_type, save_results)


def _segment_single_document(document: str, 
                           company_name: str, 
                           document_type: str,
                           save_results: bool) -> List[dict]:
    """Process a single document without chunking."""
    # Import from parent directory
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from segmentation import doc_with_lines
    
    print(f"Processing {company_name} as single document...")
    
    # Add line numbers
    document_with_lines, line2text = doc_with_lines(document)
    
    # Get structured segmentation from LLM
    structured_doc = get_structured_financial_document(document_with_lines, document_type)
    
    # Extract section text with financial metadata
    sections = get_financial_sections_text(structured_doc, line2text)
    
    # Save results if requested
    if save_results:
        try:
            save_financial_segmentation_to_json(sections, company_name, document_type)
        except Exception as e:
            print(f"Warning: Failed to save segmentation results: {e}")
    
    return sections


def _segment_chunked_document(document: str, 
                            company_name: str, 
                            document_type: str,
                            save_results: bool) -> List[dict]:
    """Process a large document using chunking strategy."""
    from .document_chunker import DocumentChunker
    
    # Import from parent directory
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from segmentation import doc_with_lines
    
    print(f"Processing {company_name} as chunked document...")
    
    # Split document into chunks
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(document)
    
    print(f"Document split into {len(chunks)} chunks for processing")
    
    all_sections = []
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        try:
            print(f"\n--- Processing Chunk {chunk.chunk_number}/{chunk.total_chunks} ---")
            print(f"Context: {chunk.heading_context}")
            print(f"Lines: {chunk.start_line}-{chunk.end_line}")
            
            # Add line numbers to chunk content
            chunk_with_lines, chunk_line2text = doc_with_lines(chunk.content)
            
            # Get structured segmentation for this chunk
            structured_chunk = get_structured_financial_document(
                chunk_with_lines, 
                document_type,
                max_retries=3
            )
            
            # Extract section text with adjusted line numbers
            chunk_sections = get_financial_sections_text(structured_chunk, chunk_line2text)
            
            # Adjust line numbers to match original document
            for section in chunk_sections:
                section['start_index'] += chunk.start_line
                section['end_index'] += chunk.start_line
                section['chunk_number'] = chunk.chunk_number
                section['chunk_context'] = chunk.heading_context
            
            all_sections.extend(chunk_sections)
            
            print(f"✅ Chunk {chunk.chunk_number} processed: {len(chunk_sections)} sections")
            
        except Exception as e:
            print(f"❌ Error processing chunk {chunk.chunk_number}: {e}")
            # Continue with other chunks even if one fails
            continue
    
    print(f"\n📋 Chunked processing complete: {len(all_sections)} total sections from {len(chunks)} chunks")
    
    # Re-sort sections by line number to ensure proper order
    all_sections.sort(key=lambda x: x['start_index'])
    
    # Recalculate character positions for the combined document
    all_sections = _recalculate_char_positions(all_sections, document)
    
    # Save results if requested
    if save_results:
        try:
            save_financial_segmentation_to_json(all_sections, company_name, document_type)
        except Exception as e:
            print(f"Warning: Failed to save segmentation results: {e}")
    
    return all_sections


def _recalculate_char_positions(sections: List[dict], original_document: str) -> List[dict]:
    """Recalculate character positions for sections based on original document."""
    lines = original_document.split('\n')
    
    for section in sections:
        # Calculate character start position
        char_start = sum(len(lines[i]) + 1 for i in range(section['start_index']))  # +1 for newline
        
        # Calculate character end position
        section_lines = lines[section['start_index']:section['end_index'] + 1]
        char_end = char_start + sum(len(line) + 1 for line in section_lines[:-1]) + len(section_lines[-1])
        
        section['char_start'] = char_start
        section['char_end'] = char_end
    
    return sections


def save_financial_segmentation_to_json(sections: List[dict], 
                                      company_name: str, 
                                      document_type: str = "10-K",
                                      output_dir: str = "segmentation_results/enterprise") -> str:
    """
    Save financial segmentation results to JSON file.
    
    Args:
        sections: List of segmented sections
        company_name: Name of the company
        document_type: Type of document
        output_dir: Directory to save results
        
    Returns:
        Path to saved JSON file
    """
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with company name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_company_name = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_company_name = safe_company_name.replace(' ', '_')
    filename = f"{safe_company_name}_{document_type}_segmented_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data for JSON serialization
    result = {
        "company_name": company_name,
        "document_type": document_type,
        "timestamp": datetime.now().isoformat(),
        "total_sections": len(sections),
        "sections": sections
    }
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Financial segmentation results saved to: {filepath}")
    return filepath