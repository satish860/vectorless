"""
Simple CUAD extractor using native OpenAI tool calling (Claude Code style).
"""

import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CUADToolExtractor:
    """CUAD extractor using native OpenAI tool calling."""
    
    def __init__(self):
        """Initialize the extractor."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError("Please set your OPENROUTER_API_KEY in the .env file")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
        self.segmentation_data = None
        self.contract_text = None
        
        # Define tools as JSON schemas
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_sections",
                    "description": "Search through document sections for relevant legal concepts using keywords",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Keywords to search for in section summaries and content"
                            },
                            "legal_concept": {
                                "type": "string",
                                "description": "Legal concept to filter by (optional)"
                            }
                        },
                        "required": ["keywords"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_section_details",
                    "description": "Get full details of a specific section by index",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "section_index": {
                                "type": "integer",
                                "description": "Index of the section to retrieve (0-based)"
                            }
                        },
                        "required": ["section_index"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_text_with_position",
                    "description": "Extract specific text and calculate its position in the original document",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "section_index": {
                                "type": "integer",
                                "description": "Index of the section containing the text"
                            },
                            "text_snippet": {
                                "type": "string", 
                                "description": "The exact text to extract and locate"
                            }
                        },
                        "required": ["section_index", "text_snippet"]
                    }
                }
            }
        ]
    
    def load_segmentation(self, filepath: str):
        """Load pre-segmented contract data."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.segmentation_data = json.load(f)
        print(f"Loaded {self.segmentation_data['total_sections']} sections from {self.segmentation_data['contract_title']}")
        
        # Reconstruct full contract text for position calculation
        self.contract_text = ""
        for section in self.segmentation_data['sections']:
            self.contract_text += section['text'] + "\n"
    
    def search_sections(self, keywords: List[str], legal_concept: str = None) -> List[Dict]:
        """Tool handler: Search sections for keywords."""
        if not self.segmentation_data:
            return {"error": "No segmentation data loaded"}
        
        matching_sections = []
        
        for idx, section in enumerate(self.segmentation_data['sections']):
            score = 0
            
            # Search in searchable_content and text
            searchable = section.get('searchable_content', '').lower()
            text = section.get('text', '').lower()
            
            # Score based on keyword matches
            for keyword in keywords:
                if keyword.lower() in searchable:
                    score += 3
                if keyword.lower() in text:
                    score += 2
            
            # Boost score for legal concept match
            if legal_concept and legal_concept.lower() in section.get('legal_concept', '').lower():
                score += 5
            
            if score > 0:
                matching_sections.append({
                    'index': idx,
                    'title': section['title'],
                    'legal_concept': section.get('legal_concept', ''),
                    'summary': section.get('summary', ''),
                    'score': score,
                    'key_terms': section.get('key_terms', [])
                })
        
        # Sort by relevance score
        matching_sections.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            "found_sections": len(matching_sections),
            "top_matches": matching_sections[:5]  # Return top 5
        }
    
    def get_section_details(self, section_index: int) -> Dict:
        """Tool handler: Get full details of a section."""
        if not self.segmentation_data or section_index < 0 or section_index >= len(self.segmentation_data['sections']):
            return {"error": f"Invalid section index {section_index}"}
        
        section = self.segmentation_data['sections'][section_index]
        return {
            "index": section_index,
            "title": section['title'],
            "legal_concept": section.get('legal_concept', ''),
            "summary": section.get('summary', ''),
            "key_terms": section.get('key_terms', []),
            "full_text": section['text'],
            "line_count": section.get('line_count', 0)
        }
    
    def extract_text_with_position(self, section_index: int, text_snippet: str) -> Dict:
        """Tool handler: Extract text and calculate position."""
        if not self.segmentation_data or section_index < 0 or section_index >= len(self.segmentation_data['sections']):
            return {"error": f"Invalid section index {section_index}"}
        
        section = self.segmentation_data['sections'][section_index]
        section_text = section['text']
        
        # Find the text in the section
        start_pos_in_section = section_text.find(text_snippet)
        if start_pos_in_section == -1:
            return {"error": f"Text snippet not found in section {section_index}"}
        
        # Calculate position in full document using char_start if available
        if 'char_start' in section:
            # Use pre-calculated character positions from segmentation
            full_doc_position = section['char_start'] + start_pos_in_section
        else:
            # Fallback: calculate manually
            full_doc_position = 0
            for i in range(section_index):
                full_doc_position += len(self.segmentation_data['sections'][i]['text']) + 1  # +1 for newline
            full_doc_position += start_pos_in_section
        
        return {
            "extracted_text": text_snippet,
            "start_position": full_doc_position,
            "section_index": section_index,
            "section_title": section['title'],
            "found": True,
            "char_start_available": 'char_start' in section
        }
    
    def handle_tool_call(self, tool_call) -> Any:
        """Handle a tool call from the LLM."""
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        if function_name == "search_sections":
            return self.search_sections(**arguments)
        elif function_name == "get_section_details":
            return self.get_section_details(**arguments)
        elif function_name == "extract_text_with_position":
            return self.extract_text_with_position(**arguments)
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    def _calculate_character_position(self, section_index: int, text: str) -> int:
        """Calculate exact character position of text in the full document."""
        if not self.segmentation_data or section_index < 0 or section_index >= len(self.segmentation_data['sections']):
            return 0
        
        section = self.segmentation_data['sections'][section_index]
        section_text = section['text']
        
        # Find the text in the section
        start_pos_in_section = section_text.find(text)
        if start_pos_in_section == -1:
            # Fallback: try to find partial match
            text_words = text.split()[:3]  # Use first 3 words
            partial_text = ' '.join(text_words)
            start_pos_in_section = section_text.find(partial_text)
            if start_pos_in_section == -1:
                return section.get('char_start', 0)  # Return section start as fallback
        
        # Calculate position in full document using char_start
        if 'char_start' in section:
            return section['char_start'] + start_pos_in_section
        else:
            # Fallback: calculate manually
            full_doc_position = 0
            for i in range(section_index):
                full_doc_position += len(self.segmentation_data['sections'][i]['text']) + 1
            return full_doc_position + start_pos_in_section
    
    def extract_clause(self, question: str, clause_type: str) -> Dict:
        """
        Extract clause using LLM with tool calling.
        
        Args:
            question: The CUAD question
            clause_type: Type of clause to extract
            
        Returns:
            Dictionary with extraction results
        """
        system_prompt = """You are a legal document analyzer that extracts specific clauses from contracts.

You have access to tools to search and extract from a pre-segmented legal document:
1. search_sections: Find sections relevant to your query
2. get_section_details: Get full text of a specific section  
3. extract_text_with_position: Extract exact text with document positions

PROCESS:
1. First, search for sections related to the clause type
2. Examine the most relevant sections in detail
3. Extract the exact text that answers the question
4. Provide precise character positions for highlighting

EXTRACTION RULES:
- For most clauses: Extract COMPLETE clauses or provisions, not just keywords
- For "Parties": Extract specific party names, entity types, and identifiers (not full clauses)
- Include full sentences that contain the legal provision  
- Get exact character positions in the original document
- There may be multiple instances of the same clause type
- Each answer should be a distinct, specific text span

When you have found the answer, respond ONLY with valid JSON in this exact format:
{
  "answers": [{"text": "exact extracted text", "section_index": section_number}],
  "is_impossible": false,
  "reasoning": "step-by-step explanation"
}

If the clause doesn't exist, use:
{
  "answers": [],
  "is_impossible": true,
  "reasoning": "explanation why not found"
}

Note: Only provide the extracted text and section_index. We will calculate the exact character position."""

        user_prompt = f"""Question: {question}

Clause Type: {clause_type}

{"SPECIAL INSTRUCTION for Parties: Extract each individual party identifier separately - company names, defined terms like \"Company\" or \"Distributor\", entity names, etc. Each should be a separate answer. Look throughout the document including notices/contact sections. Extract just the specific name/identifier, not surrounding text." if clause_type == "Parties" else ""}

Use the available tools to search the document, examine relevant sections, and extract the exact text that answers this question."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"LLM iteration {iteration}")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                
                # Check if LLM wants to call tools
                if message.tool_calls:
                    # Add the assistant's message
                    messages.append(message)
                    
                    # Handle each tool call
                    for tool_call in message.tool_calls:
                        print(f"LLM called tool: {tool_call.function.name}")
                        print(f"Arguments: {tool_call.function.arguments}")
                        
                        result = self.handle_tool_call(tool_call)
                        
                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, indent=2)
                        })
                
                else:
                    # LLM provided final answer
                    print("LLM provided final answer")
                    content = message.content
                    print(f"Raw LLM content: '{content}'")
                    print(f"Content length: {len(content)}")
                    
                    if not content or content.strip() == "":
                        print("Empty response from LLM")
                        return {
                            "answers": [],
                            "is_impossible": True,
                            "reasoning": "Empty response from LLM"
                        }
                    
                    # Clean up content if it has extra text before JSON
                    if '{' in content:
                        json_start = content.find('{')
                        content = content[json_start:]
                    
                    try:
                        # Try to parse as JSON
                        answer = json.loads(content)
                        
                        # Calculate accurate character positions for each answer
                        if answer.get('answers'):
                            for ans in answer['answers']:
                                if 'section_index' in ans and 'text' in ans:
                                    pos = self._calculate_character_position(ans['section_index'], ans['text'])
                                    ans['answer_start'] = pos
                                    # Remove section_index as it's no longer needed
                                    del ans['section_index']
                        
                        return answer
                    except json.JSONDecodeError as e:
                        # Fallback if not valid JSON
                        print(f"JSON parse error: {e}")
                        print(f"Content to parse: '{content}'")
                        return {
                            "answers": [],
                            "is_impossible": True,
                            "reasoning": f"Could not parse LLM response: {content}"
                        }
                
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                return {
                    "answers": [],
                    "is_impossible": True,
                    "reasoning": f"Error during extraction: {str(e)}"
                }
        
        return {
            "answers": [],
            "is_impossible": True,
            "reasoning": f"Maximum iterations ({max_iterations}) reached without final answer"
        }


def test_both_scenarios():
    """Test both answer present and impossible scenarios."""
    extractor = CUADToolExtractor()
    
    # Load segmentation
    segmentation_file = r"C:\Source\vectorless\segmentation_results\LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR_AGREEMEN_cached.json"
    extractor.load_segmentation(segmentation_file)
    
    # Test 1: Question with answer present (Governing Law)
    print("\n" + "=" * 80)
    print("TEST 1: ANSWER PRESENT - Governing Law")
    print("=" * 80)
    
    question1 = "Highlight the parts (if any) of this contract related to \"Governing Law\" that should be reviewed by a lawyer. Details: Which state/country's law governs the interpretation of the contract?"
    clause_type1 = "Governing Law"
    
    result1 = extractor.extract_clause(question1, clause_type1)
    
    print(f"Found {len(result1.get('answers', []))} answer(s)")
    print(f"Is Impossible: {result1.get('is_impossible', False)}")
    print(f"Reasoning: {result1.get('reasoning', 'No reasoning provided')[:200]}...")
    
    if result1.get('answers'):
        for i, answer in enumerate(result1['answers'], 1):
            print(f"Answer {i}: Position {answer.get('answer_start', 'unknown')}")
            print(f"  Text: {answer.get('text', 'No text')[:100]}...")
    
    # Test 2: Question with no answer (Revenue/Profit Sharing)
    print("\n" + "=" * 80)
    print("TEST 2: NO ANSWER (IMPOSSIBLE) - Revenue/Profit Sharing")
    print("=" * 80)
    
    question2 = "Highlight the parts (if any) of this contract related to \"Revenue/Profit Sharing\" that should be reviewed by a lawyer. Details: Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?"
    clause_type2 = "Revenue/Profit Sharing"
    
    result2 = extractor.extract_clause(question2, clause_type2)
    
    print(f"Found {len(result2.get('answers', []))} answer(s)")
    print(f"Is Impossible: {result2.get('is_impossible', False)}")
    print(f"Reasoning: {result2.get('reasoning', 'No reasoning provided')[:200]}...")
    
    if result2.get('answers'):
        for i, answer in enumerate(result2['answers'], 1):
            print(f"Answer {i}: Position {answer.get('answer_start', 'unknown')}")
            print(f"  Text: {answer.get('text', 'No text')[:100]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Test 1 (Governing Law): {'SUCCESS' if not result1.get('is_impossible', True) and result1.get('answers') else 'FAILED'}")
    print(f"Test 2 (Revenue Sharing): {'SUCCESS' if result2.get('is_impossible', False) and not result2.get('answers') else 'FAILED'}")
    
    return result1, result2


if __name__ == "__main__":
    test_both_scenarios()