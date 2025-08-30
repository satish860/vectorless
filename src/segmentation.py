"""
Document segmentation module using instructor and OpenRouter GPT-OSS-120B.
Adapted from the instructor library's document segmentation example.
"""

import os
import time
import json
from typing import List
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Section(BaseModel):
    """Represents a meaningful section of a document with enhanced searchability."""
    title: str = Field(description="Main topic of this section of the document")
    start_index: int = Field(description="Line number where the section begins")
    end_index: int = Field(description="Line number where the section ends")
    legal_concept: str = Field(description="Primary legal area or concept this section covers")
    summary: str = Field(description="2-3 sentence summary including key provisions and obligations")
    key_terms: List[str] = Field(description="List of important searchable terms and keywords from this section")
    reasoning: str = Field(description="Chain of thought reasoning about why this section was identified and what it contains")


class StructuredDocument(BaseModel):
    """Document with meaningful sections, each centered around a single concept/topic."""
    sections: List[Section] = Field(description="A list of sections of the document")


def doc_with_lines(document: str) -> tuple[str, dict[int, str]]:
    """
    Add line numbers to document and create line mapping.
    
    Args:
        document: The original document text
        
    Returns:
        tuple: (document_with_line_numbers, line2text_mapping)
    """
    document_lines = document.split("\n")
    document_with_line_numbers = ""
    line2text = {}
    
    for i, line in enumerate(document_lines):
        document_with_line_numbers += f"[{i}] {line}\n"
        line2text[i] = line
    
    return document_with_line_numbers, line2text


def create_instructor_client():
    """Create and configure instructor client for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"API Key loaded: {'Yes' if api_key and api_key != 'your_api_key_here' else 'No'}")
    
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("Please set your OPENROUTER_API_KEY in the .env file")
    
    client = instructor.from_openai(
        OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        ),
        mode=instructor.Mode.JSON
    )
    
    return client


def get_structured_document(document_with_line_numbers: str, max_retries: int = 3) -> StructuredDocument:
    """
    Use LLM to segment document into meaningful sections with retry logic.
    
    Args:
        document_with_line_numbers: Document text with line numbers
        max_retries: Maximum number of retry attempts
        
    Returns:
        StructuredDocument with identified sections
    """
    client = create_instructor_client()
    model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
    
    print(f"Using model: {model_name}")
    print(f"Document length: {len(document_with_line_numbers)} characters")
    
    system_prompt = """You are a world class legal document analyzer working on organizing contract sections with enhanced searchability.

Your task is to create a StructuredDocument with detailed sections that include summaries and searchable keywords.

CHAIN OF THOUGHT PROCESS:
For each section, think step by step:

Step 1: IDENTIFY the boundaries - Where does one legal concept end and another begin?
Step 2: ANALYZE the legal concept - What area of law does this section cover?
Step 3: EXTRACT key provisions - What are the main obligations, rights, conditions, or requirements?
Step 4: IDENTIFY searchable terms - What keywords would a lawyer or QA system search for to find this content?
Step 5: SYNTHESIZE into summary - Create a 2-3 sentence summary that includes the key legal terms and provisions.

INPUT DOCUMENT FORMAT:
Each line is marked with its line number in square brackets (e.g. [1], [2], [3], etc). Use these line numbers for start_index and end_index.

SECTION CATEGORIES to consider:
- Contract identification and parties
- Terms and duration  
- Obligations and responsibilities
- Payment and financial terms
- Termination clauses
- Liability and limitations
- Governing law and jurisdiction
- Warranties and representations
- Confidentiality and non-disclosure
- Force majeure and exceptions
- Other distinct legal provisions

For each section, provide:
- title: Clear, descriptive title of the legal concept
- legal_concept: The primary area of law (e.g., "Contract Formation", "Termination Rights", "Payment Obligations")
- summary: 2-3 sentences describing key provisions with searchable legal terms
- key_terms: Array of important keywords that would be used in searches (e.g., ["termination", "30-day notice", "material breach", "cure period"])
- reasoning: Brief explanation of why you identified this as a distinct section and what makes it legally significant

EXAMPLE REASONING:
"This section covers termination provisions because it establishes specific conditions under which either party can end the agreement. The 30-day notice requirement and material breach language are standard termination mechanisms that lawyers would search for when reviewing exit strategies."

Make sections meaningful, self-contained, and optimized for keyword searching."""
    
    for attempt in range(max_retries):
        try:
            print(f"Sending request to OpenRouter... (Attempt {attempt + 1}/{max_retries})")
            response = client.chat.completions.create(
                model=model_name,
                response_model=StructuredDocument,
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


def get_sections_text(structured_doc: StructuredDocument, line2text: dict[int, str]) -> List[dict]:
    """
    Extract the actual text content for each section with enhanced metadata.
    
    Args:
        structured_doc: The structured document with sections
        line2text: Mapping from line numbers to text
        
    Returns:
        List of dictionaries containing section info, text, and searchable metadata
    """
    sections = []
    
    for section in structured_doc.sections:
        # Extract text from start_index to end_index
        section_lines = []
        for line_num in range(section.start_index, min(section.end_index + 1, len(line2text))):
            if line_num in line2text:
                section_lines.append(line2text[line_num])
        
        section_text = "\n".join(section_lines)
        
        sections.append({
            "title": section.title,
            "start_index": section.start_index,
            "end_index": section.end_index,
            "text": section_text,
            "line_count": section.end_index - section.start_index + 1,
            "legal_concept": section.legal_concept,
            "summary": section.summary,
            "key_terms": section.key_terms,
            "reasoning": section.reasoning,
            # Create searchable content combining title, summary, and key terms for grep
            "searchable_content": f"{section.title} | {section.legal_concept} | {section.summary} | {' '.join(section.key_terms)}"
        })
    
    return sections


def save_segmentation_to_json(sections: List[dict], contract_title: str, output_dir: str = "segmentation_results") -> str:
    """
    Save segmentation results to JSON file.
    
    Args:
        sections: List of segmented sections
        contract_title: Title of the contract
        output_dir: Directory to save results
        
    Returns:
        Path to saved JSON file
    """
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"segmentation_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data for JSON serialization
    result = {
        "contract_title": contract_title,
        "timestamp": datetime.now().isoformat(),
        "total_sections": len(sections),
        "sections": sections
    }
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Segmentation results saved to: {filepath}")
    return filepath


def load_segmentation_from_json(filepath: str) -> dict:
    """
    Load segmentation results from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with segmentation data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Segmentation file not found: {filepath}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {filepath}")


def segment_document(document: str, save_results: bool = True) -> List[dict]:
    """
    Complete pipeline to segment a document into meaningful sections.
    
    Args:
        document: The original document text
        save_results: Whether to save results to JSON file
        
    Returns:
        List of section dictionaries with titles, indices, and text content
    """
    # Add line numbers
    document_with_lines, line2text = doc_with_lines(document)
    
    # Get structured segmentation from LLM
    structured_doc = get_structured_document(document_with_lines)
    
    # Extract section text
    sections = get_sections_text(structured_doc, line2text)
    
    # Save results if requested
    if save_results:
        try:
            # Use a generic title since we don't have contract title at this level
            save_segmentation_to_json(sections, "Document_Segmentation")
        except Exception as e:
            print(f"Warning: Failed to save segmentation results: {e}")
    
    return sections