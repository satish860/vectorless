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
                    "name": "multi_grep_search",
                    "description": "Search multiple regex patterns at once - much more efficient than multiple grep_search calls",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patterns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Array of regex patterns to search for simultaneously"
                            },
                            "context_lines": {
                                "type": "integer",
                                "description": "Number of context lines before and after each match (default: 2)"
                            },
                            "case_sensitive": {
                                "type": "boolean",
                                "description": "Whether search should be case sensitive (default: false)"
                            }
                        },
                        "required": ["patterns"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_multiple_sections",
                    "description": "Get full details of multiple sections at once by their indices",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "section_indices": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Array of section indices to retrieve (0-based)"
                            }
                        },
                        "required": ["section_indices"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "multi_extract_text",
                    "description": "Extract multiple text snippets with positions at once - much more efficient",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "extractions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "section_index": {
                                            "type": "integer",
                                            "description": "Index of section containing the text"
                                        },
                                        "text_snippet": {
                                            "type": "string",
                                            "description": "The exact text to extract and locate"
                                        }
                                    },
                                    "required": ["section_index", "text_snippet"]
                                },
                                "description": "Array of text extraction requests"
                            }
                        },
                        "required": ["extractions"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document_overview",
                    "description": "Get overview of all sections with titles, legal concepts, and summaries for strategic planning",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
        ]
    
    def _get_document_structure_priority(self, clause_type: str) -> list:
        """Map clause types to priority document sections (lawyer's approach)"""
        structure_map = {
            "Agreement Date": ["preamble", "opening_recital", "signature_blocks", "definitions"],
            "Effective Date": ["term_sections", "commencement", "conditions_precedent", "opening"],
            "Expiration Date": ["term_sections", "duration", "termination", "renewal"],
            "Governing Law": ["general_provisions", "miscellaneous", "boilerplate", "final_sections"],
            "Parties": ["preamble", "opening_recital", "definitions", "notices"],
            "Document Name": ["title", "header", "preamble", "integration_clause"],
            
            "Exclusivity": ["grant_section", "appointment", "rights_section", "early_clauses"],
            "Minimum Commitment": ["performance_obligations", "purchase_requirements", "financial_terms"],
            "Price Restrictions": ["pricing_sections", "payment_terms", "financial_provisions"],
            "Termination For Convenience": ["termination_section", "default_provisions"],
            
            "License Grant": ["grant_section", "rights_section", "intellectual_property"],
            "Non-Transferable License": ["license_terms", "assignment_restrictions", "transfer_provisions"],
            "Warranty Duration": ["warranties_section", "representations", "product_terms"],
            "Insurance": ["risk_allocation", "insurance_requirements", "liability_sections"],
            
            "Non-Compete": ["restrictive_covenants", "post_employment", "competition_restrictions"],
            "Change Of Control": ["corporate_events", "assignment_section", "control_provisions"],
            "Anti-Assignment": ["assignment_section", "transfer_restrictions", "general_provisions"]
        }
        return structure_map.get(clause_type, ["search_all_sections"])
    
    def _get_professional_checklist(self, clause_type: str) -> list:
        """Professional validation checklist for each clause type"""
        checklists = {
            "Exclusivity": [
                "✓ Does it grant EXCLUSIVE rights (not just general rights)?",
                "✓ Is there geographic or market scope defined?", 
                "✓ Are there restrictions on sourcing from others?",
                "✓ Does it prevent dealing with competitors?"
            ],
            "Non-Compete": [
                "✓ Does it restrict COMPETING business activities (not just exclusivity)?",
                "✓ Is there a defined scope (geographic/temporal/industry)?",
                "✓ Does it prevent engaging in competitive business?",
                "✓ Is it different from exclusive dealing arrangements?"
            ],
            "Minimum Commitment": [
                "✓ Are there specific minimum quantities/amounts required?",
                "✓ Is there a time period specified (annual/quarterly)?",
                "✓ Are there consequences for failing to meet minimums?",
                "✓ Is it mandatory (not just targets or forecasts)?"
            ],
            "Non-Transferable License": [
                "✓ Does it explicitly state 'non-transferable' or 'cannot transfer'?",
                "✓ Are there restrictions on sublicensing or assignment?",
                "✓ Is it different from general license grants?",
                "✓ Does it prohibit specific transfer activities?"
            ],
            "Price Restrictions": [
                "✓ Are there specific pricing controls or limitations?",
                "✓ Does it restrict pricing flexibility?",
                "✓ Are there maximum/minimum price requirements?",
                "✓ Does it control pricing methodology?"
            ],
            "Cap On Liability": [
                "✓ Is there an explicit monetary cap or limit?",
                "✓ Does it use language like 'limited to', 'not exceed', 'maximum'?",
                "✓ Is it different from general indemnification?",
                "✓ Does it set specific liability limits?"
            ]
        }
        return checklists.get(clause_type, [])
    
    def _get_professional_search_patterns(self, clause_type: str) -> dict:
        """Professional search patterns for systematic clause extraction"""
        search_patterns = {
            "Exclusivity": {
                "primary_keywords": ["exclusive", "exclusively", "sole", "only"],
                "secondary_keywords": ["distributor", "dealer", "territory", "license", "rights"],
                "context_phrases": ["exclusive rights", "sole distributor", "exclusive territory", "exclusive license"],
                "search_strategy": "Look for combinations of exclusivity terms with business relationship terms",
                "validation_approach": "Check for geographic, product, or customer scope limitations"
            },
            "Minimum Commitment": {
                "primary_keywords": ["minimum", "at least", "not less than", "shall purchase"],
                "secondary_keywords": ["purchase", "order", "volume", "quantity", "commitment"],
                "context_phrases": ["minimum purchase", "minimum order", "minimum volume", "commit to purchase"],
                "search_strategy": "Look for specific quantities, amounts, or time-bound commitments",
                "validation_approach": "Verify numerical amounts and enforcement obligations"
            },
            "Price Restrictions": {
                "primary_keywords": ["price", "pricing", "resale", "retail"],
                "secondary_keywords": ["minimum", "maximum", "restriction", "control", "guideline"],
                "context_phrases": ["pricing restrictions", "resale price", "minimum pricing", "price control"],
                "search_strategy": "Search for pricing constraints, guidelines, or market controls",
                "validation_approach": "Confirm impact on pricing flexibility and competitive behavior"
            },
            "Agreement Date": {
                "primary_keywords": ["dated", "entered into", "made", "executed"],
                "secondary_keywords": ["day of", "as of", "this", "agreement"],
                "context_phrases": ["entered into as of", "made and entered", "this agreement dated"],
                "search_strategy": "Look in opening paragraphs, preambles, and signature blocks",
                "validation_approach": "Extract actual dates, not references to dates"
            },
            "Non-Compete": {
                "primary_keywords": ["compete", "competition", "competitive", "non-compete"],
                "secondary_keywords": ["restrict", "prohibit", "covenant", "business"],
                "context_phrases": ["non-compete", "non-competition", "compete with", "competitive business"],
                "search_strategy": "Distinguish from exclusivity - look for business activity restrictions",
                "validation_approach": "Confirm restriction on competitive business activities, not just sourcing"
            }
        }
        return search_patterns.get(clause_type, {
            "primary_keywords": [],
            "secondary_keywords": [],
            "context_phrases": [],
            "search_strategy": "Apply general systematic search approach",
            "validation_approach": "Use professional judgment and conservative standards"
        })
    
    def _get_clause_guidance(self, clause_type: str) -> str:
        """Get specific guidance for different clause types."""
        guidance = {
            # FACTUAL CLAUSES - Liberal extraction threshold (Lawyer's Review Pattern)
            "Agreement Date": """
FACTUAL CLAUSE - LAWYER'S REVIEW APPROACH:
- PRIMARY TARGET: Opening recital date: "made and entered into as of [DATE]"
- SECONDARY: Signature dates in closing/execution sections  
- TERTIARY: Defined "Execution Date" or "Agreement Date" terms
- LEGAL SIGNIFICANCE: Establishes contract formation date for statute of limitations, retroactive analysis
- SEARCH LOCATIONS: Opening paragraph, preamble, signature blocks, definitions
- EXTRACT ANY: Date showing when parties agreed to be bound (execution/signature date)
- COMMON LAWYER PHRASES: "dated", "made on", "entered into as of", "this [X] day of [MONTH]", "executed on"
- FORMATS: "September 7, 1999", "7th day of September, 1999", "as of September 7, 1999"
""",
            
            "Expiration Date": """
FACTUAL CLAUSE - LAWYER'S REVIEW APPROACH:
- PRIMARY: Explicit expiration dates: "expires on [DATE]", "shall terminate on [DATE]"
- SECONDARY: Calculate from term + commencement: "ten (10) years from [START DATE]"
- TERTIARY: End of initial term before renewals
- LEGAL SIGNIFICANCE: When contract legally ends, renewal deadlines, notice periods
- SEARCH LOCATIONS: Term sections, duration clauses, renewal provisions
- EXTRACT WHEN: Clear end date or calculable term duration is specified
- COMMON PHRASES: "term of", "expires", "terminates", "initial term", "unless renewed"
- NOTE: Term duration ("10 years") counts as expiration date information for legal review
""",
            
            "Effective Date": """
FACTUAL CLAUSE - LAWYER'S REVIEW APPROACH:
- PRIMARY: "Effective" language: "effective as of", "effective immediately", "effective upon"
- SECONDARY: Commencement dates: "commencing on", "shall commence"  
- DISTINGUISH: Different from Agreement Date if conditions precedent exist
- LEGAL SIGNIFICANCE: When obligations actually begin, retroactive vs prospective effect
- SEARCH LOCATIONS: Opening sections, term clauses, conditions precedent
- COMMON PHRASES: "effective", "commence", "begin", "start", "take effect"
- CONDITIONAL: "effective upon delivery", "subject to conditions", "upon satisfaction of"
""",
            
            # CONCEPTUAL CLAUSES - High threshold, prevent confusion
            "Non-Compete": """
CONCEPTUAL CLAUSE - STRICT REQUIREMENTS:
- MUST restrict competing business activities or competition itself
- CRITICAL: Exclusive distributor rights ≠ Non-compete restrictions
- Exclusivity (can't buy from others) ≠ Non-compete (can't compete in business)
- Look for: "shall not compete", "non-competition", "restraint on competition"
- Mark IMPOSSIBLE if only exclusivity/sourcing restrictions found
""",
            
            "Change Of Control": """
CONCEPTUAL CLAUSE - STRICT REQUIREMENTS:
- MUST have triggers for ownership/control changes (merger, acquisition, stock sale)
- Assignment restrictions ≠ Change of control provisions
- Look for: "change in control", "merger", "acquisition", "sale of assets"
- Mark IMPOSSIBLE if only general assignment restrictions found
""",
            
            "Non-Transferable License": """
CONCEPTUAL CLAUSE - Must explicitly restrict transfer:
- MUST explicitly state "non-transferable", "may not transfer", "cannot assign"
- Basic license grants ≠ Non-transferable licenses
- Look for explicit transfer prohibitions in license terms
- Mark IMPOSSIBLE if just general license without transfer restrictions
""",
            
            "Unlimited/All-You-Can-Eat-License": """
CONCEPTUAL CLAUSE - Must explicitly state unlimited scope:
- MUST explicitly state "unlimited", "unrestricted", "all-you-can-eat"
- Basic license grants ≠ Unlimited licenses
- Look for language indicating no usage limits or restrictions
- Mark IMPOSSIBLE if just standard license without unlimited language
""",
            
            # EXISTING CLAUSES
            "Most Favored Nation": """
Mark IMPOSSIBLE unless explicit preferential treatment:
- Must state "better terms than other parties" or similar preferential language
- General pricing or rate adjustments ≠ Most Favored Nation
- Look for: "most favored", "preferential treatment", "better terms"
""",
            
            "No-Solicit Of Customers": """
Mark IMPOSSIBLE unless explicit customer solicitation restrictions:
- Must restrict approaching/targeting the other party's customers
- Exclusivity agreements ≠ customer non-solicitation
- Look for: "solicit customers", "approach customers", "target clients"
""",
            
            "Minimum Commitment": """
PERFORMANCE CLAUSE - LAWYER'S SYSTEMATIC APPROACH:
- PRIMARY TARGETS: "minimum purchase", "minimum order", "minimum volume", "shall purchase not less than"
- SECONDARY: Volume commitments: "at least [X] units", "minimum of [X] annually", "commit to purchase"
- TERTIARY: Financial minimums: "minimum spend of", "guarantee minimum revenue", "minimum fees"
- LEGAL SIGNIFICANCE: Enforceable performance obligations, breach consequences
- SEARCH LOCATIONS: Performance sections, purchasing clauses, volume requirements, commitment terms
- COMMON PHRASES: "minimum", "not less than", "at least", "shall purchase", "commit to", "guarantee to order"
- FORMATS: Specific numbers (units, dollars, percentages), time-bound commitments (annual, monthly)
- PROFESSIONAL TIP: Look for both purchase minimums AND usage/distribution minimums
""",
            
            "Ip Ownership Assignment": """
Mark IMPOSSIBLE unless IP ownership is explicitly assigned:
- Must transfer IP ownership from one party to another
- Basic license grants ≠ ownership assignment
- Look for: "assigns", "transfers ownership", "hereby assigns"
""",
            
            "Irrevocable Or Perpetual License": """
Mark IMPOSSIBLE unless explicitly stated as permanent:
- Must use words like "irrevocable", "perpetual", "permanent", "in perpetuity"
- Basic license grants with terms ≠ irrevocable/perpetual
- Term-limited licenses ≠ perpetual licenses
""",
            
            "Post-Termination Services": """
Mark FOUND when services continue after contract ends:
- Look for obligations/services that survive termination
- Search: termination sections, survival clauses, post-termination duties
- Can include: support, maintenance, transition services
""",
            
            "Audit Rights": """
Mark IMPOSSIBLE unless explicit audit/inspection rights:
- Must give one party right to examine the other's records/operations
- General reporting ≠ audit rights
- Look for: "audit", "inspect", "examine records", "right to inspect"
""",
            
            "Warranty Duration": """
Mark FOUND when warranty time periods are specified:
- Look for specific time periods for warranties/guarantees
- Search: warranty sections, guarantee periods, "for a period of"
- Can be months, years, or other timeframes
""",
            
            "Third Party Beneficiary": """
Mark IMPOSSIBLE unless explicit third party rights:
- Must explicitly state third parties can enforce or benefit
- General contract language ≠ third party beneficiary rights
- Look for: "third party beneficiary", "third parties may enforce"
""",
            
            "Exclusivity": """
BUSINESS RELATIONSHIP CLAUSE - LAWYER'S COMPREHENSIVE APPROACH:
- PRIMARY TARGETS: "exclusive", "exclusively", "sole", "only", "exclusive rights"
- SECONDARY: "exclusive distributor", "exclusive dealer", "exclusive territory", "exclusive license" 
- TERTIARY: Restrictions on others: "shall not appoint other", "no other distributors", "sole source"
- LEGAL SIGNIFICANCE: Monopolistic rights, competitive protection, market control
- SEARCH LOCATIONS: Grant sections, appointment clauses, distribution rights, territory definitions
- COMMON PHRASES: "exclusive rights", "exclusive territory", "sole distributor", "exclusive license"
- PROFESSIONAL APPROACH: Look for 3 types:
  1. TERRITORIAL EXCLUSIVITY: Geographic market protection
  2. PRODUCT EXCLUSIVITY: Exclusive rights to specific products/services
  3. CUSTOMER EXCLUSIVITY: Exclusive access to certain customers/segments
- CRITICAL: Even limited exclusivity (by geography, product, or customer) counts as exclusivity
""",
            
            "Price Restrictions": """
COMMERCIAL TERMS CLAUSE - LAWYER'S FINANCIAL REVIEW APPROACH:
- PRIMARY TARGETS: "pricing restrictions", "price controls", "pricing guidelines", "price maintenance"
- SECONDARY: "minimum pricing", "maximum pricing", "price floors", "price ceilings", "resale price"
- TERTIARY: "suggested retail price", "MAP pricing", "pricing policies", "discount restrictions"
- LEGAL SIGNIFICANCE: Antitrust implications, pricing freedom limitations, resale restrictions
- SEARCH LOCATIONS: Pricing sections, commercial terms, resale provisions, dealer agreements
- COMMON PHRASES: "pricing", "price", "resale", "retail price", "discount", "markup", "margin"
- PROFESSIONAL CATEGORIES:
  1. MINIMUM PRICE RESTRICTIONS: Price floors, minimum resale prices
  2. MAXIMUM PRICE RESTRICTIONS: Price ceilings, maximum retail prices  
  3. PRICING GUIDELINES: Suggested pricing, recommended prices, pricing policies
- ANTI-COMPETITIVE FOCUS: Any language limiting pricing freedom or setting price parameters
"""
        }
        
        return guidance.get(clause_type, "Apply general conservative approach - mark impossible unless explicit provision exists.")
    
    def load_segmentation(self, filepath: str):
        """Load pre-segmented contract data."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.segmentation_data = json.load(f)
        print(f"Loaded {self.segmentation_data['total_sections']} sections from {self.segmentation_data['contract_title']}")
        
        # Reconstruct full contract text for position calculation
        self.contract_text = ""
        for section in self.segmentation_data['sections']:
            self.contract_text += section['text'] + "\n"
    
    def multi_grep_search(self, patterns: List[str], context_lines: int = 2, case_sensitive: bool = False) -> Dict:
        """Tool handler: Search contract text using multiple regex patterns simultaneously."""
        if not self.contract_text:
            return {"error": "No contract text loaded"}
        
        import re
        
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            compiled_patterns = []
            
            # Compile all patterns
            for pattern in patterns:
                try:
                    compiled_patterns.append((pattern, re.compile(pattern, flags)))
                except re.error as e:
                    return {"error": f"Invalid regex pattern '{pattern}': {e}"}
            
            lines = self.contract_text.split('\n')
            all_matches = []
            pattern_stats = {}
            
            # Search each line against all patterns
            for i, line in enumerate(lines):
                for pattern, regex in compiled_patterns:
                    if regex.search(line):
                        # Track pattern usage
                        if pattern not in pattern_stats:
                            pattern_stats[pattern] = 0
                        pattern_stats[pattern] += 1
                        
                        # Get context lines
                        start_line = max(0, i - context_lines)
                        end_line = min(len(lines), i + context_lines + 1)
                        
                        context = []
                        for j in range(start_line, end_line):
                            prefix = ">>> " if j == i else "    "
                            context.append(f"{prefix}{lines[j]}")
                        
                        # Calculate character position
                        char_pos = sum(len(lines[k]) + 1 for k in range(i))  # +1 for newline
                        
                        all_matches.append({
                            "line_number": i + 1,
                            "char_position": char_pos,
                            "matching_line": line.strip(),
                            "context": context,
                            "pattern_used": pattern
                        })
            
            # Sort by character position
            all_matches.sort(key=lambda x: x["char_position"])
            
            return {
                "patterns_searched": patterns,
                "total_matches_found": len(all_matches),
                "matches_per_pattern": pattern_stats,
                "matches": all_matches[:15]  # Limit to first 15 matches across all patterns
            }
            
        except Exception as e:
            return {"error": f"Search error: {e}"}
    
    def get_multiple_sections(self, section_indices: List[int]) -> Dict:
        """Tool handler: Get full details of multiple sections at once."""
        if not self.segmentation_data:
            return {"error": "No segmentation data loaded"}
        
        sections_data = []
        errors = []
        
        for idx in section_indices:
            if idx < 0 or idx >= len(self.segmentation_data['sections']):
                errors.append(f"Invalid section index {idx}")
                continue
            
            section = self.segmentation_data['sections'][idx]
            sections_data.append({
                "index": idx,
                "title": section['title'],
                "legal_concept": section.get('legal_concept', ''),
                "summary": section.get('summary', ''),
                "key_terms": section.get('key_terms', []),
                "full_text": section['text'],
                "line_count": section.get('line_count', 0)
            })
        
        result = {
            "requested_indices": section_indices,
            "sections_retrieved": len(sections_data),
            "sections": sections_data
        }
        
        if errors:
            result["errors"] = errors
        
        return result
    
    def multi_extract_text(self, extractions: List[Dict]) -> Dict:
        """Tool handler: Extract multiple text snippets with positions at once."""
        if not self.segmentation_data:
            return {"error": "No segmentation data loaded"}
        
        results = []
        errors = []
        
        for i, extraction in enumerate(extractions):
            section_index = extraction.get("section_index")
            text_snippet = extraction.get("text_snippet")
            
            # Validate inputs
            if section_index is None or text_snippet is None:
                errors.append(f"Extraction {i}: Missing section_index or text_snippet")
                continue
                
            if section_index < 0 or section_index >= len(self.segmentation_data['sections']):
                errors.append(f"Extraction {i}: Invalid section index {section_index}")
                continue
            
            section = self.segmentation_data['sections'][section_index]
            section_text = section['text']
            
            # Find the text in the section
            start_pos_in_section = section_text.find(text_snippet)
            if start_pos_in_section == -1:
                errors.append(f"Extraction {i}: Text snippet '{text_snippet[:50]}...' not found in section {section_index}")
                continue
            
            # Calculate position in full document
            if 'char_start' in section:
                full_doc_position = section['char_start'] + start_pos_in_section
            else:
                # Fallback: calculate manually  
                full_doc_position = 0
                for j in range(section_index):
                    full_doc_position += len(self.segmentation_data['sections'][j]['text']) + 1
                full_doc_position += start_pos_in_section
            
            results.append({
                "extraction_index": i,
                "extracted_text": text_snippet,
                "start_position": full_doc_position,
                "section_index": section_index,
                "section_title": section['title'],
                "found": True
            })
        
        # Sort results by document position
        results.sort(key=lambda x: x["start_position"])
        
        response = {
            "total_extractions_requested": len(extractions),
            "successful_extractions": len(results),
            "failed_extractions": len(errors),
            "results": results
        }
        
        if errors:
            response["errors"] = errors
            
        return response
    
    def get_document_overview(self) -> Dict:
        """Tool handler: Get overview of all sections for strategic planning."""
        if not self.segmentation_data:
            return {"error": "No segmentation data loaded"}
        
        sections = self.segmentation_data['sections']
        overview = {
            "contract_title": self.segmentation_data.get('contract_title', 'Unknown Contract'),
            "total_sections": len(sections),
            "sections_overview": []
        }
        
        for i, section in enumerate(sections):
            overview["sections_overview"].append({
                "index": i,
                "title": section.get('title', f'Section {i}'),
                "legal_concept": section.get('legal_concept', 'Unknown'),
                "summary": section.get('summary', 'No summary available')[:200] + ("..." if len(section.get('summary', '')) > 200 else ""),
                "key_terms": section.get('key_terms', [])[:5],  # Limit to first 5 key terms
                "line_count": section.get('line_count', 0)
            })
        
        return overview
    
    def handle_tool_call(self, tool_call) -> Any:
        """Handle a tool call from the LLM."""
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        if function_name == "multi_grep_search":
            return self.multi_grep_search(**arguments)
        elif function_name == "get_multiple_sections":
            return self.get_multiple_sections(**arguments)
        elif function_name == "multi_extract_text":
            return self.multi_extract_text(**arguments)
        elif function_name == "get_document_overview":
            return self.get_document_overview()
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
        system_prompt = f"""You are a legal document analyzer specializing in contract clause extraction. Your task is to systematically analyze contracts and extract specific legal provisions with high precision.

## TASK DECOMPOSITION AND PLANNING

Your analysis MUST follow this systematic approach:

### PHASE 1: STRATEGIC DOCUMENT ANALYSIS
1. **ALWAYS** start by calling get_document_overview() to understand the contract structure
2. **PLAN EXTENSIVELY**: Identify the 2-4 most relevant sections based on:
   - Legal concept alignment with target clause type
   - Key terms that directly relate to the query
   - Summary content that indicates clause presence
3. **REFLECT**: Confirm your section selection addresses the user's specific request

### PHASE 2: TARGETED EXTRACTION 
4. Use get_multiple_sections() to examine your identified sections in detail
5. Extract precise text using multi_extract_text() from the most relevant sections
6. **FALLBACK ONLY**: Use multi_grep_search() if targeted approach yields no results

### PHASE 3: VALIDATION AND COMPLETION
7. **REFLECT EXTENSIVELY** on whether each extracted text directly answers the query
8. Confirm that you have completely resolved the user's clause extraction request
9. Do NOT guess or make assumptions - if uncertain, mark as impossible

## PROFESSIONAL DOCUMENT STRUCTURE APPROACH FOR "{clause_type}":

**PRIORITY SECTIONS** (where lawyers look first):
{', '.join(self._get_document_structure_priority(clause_type))}

**PROFESSIONAL VALIDATION CHECKLIST**:
{chr(10).join(self._get_professional_checklist(clause_type)) if self._get_professional_checklist(clause_type) else "Apply general professional judgment"}

**PROFESSIONAL SEARCH PATTERNS** (systematic keyword approach):
Primary Keywords: {', '.join(self._get_professional_search_patterns(clause_type).get('primary_keywords', []))}
Context Phrases: {', '.join(self._get_professional_search_patterns(clause_type).get('context_phrases', []))}
Search Strategy: {self._get_professional_search_patterns(clause_type).get('search_strategy', 'Apply general approach')}

## CLAUSE-SPECIFIC ANALYSIS FOR "{clause_type}":

{self._get_clause_guidance(clause_type)}

## EXTRACTION THRESHOLDS BY CLAUSE CATEGORY

Apply different standards based on clause type:

**FACTUAL CLAUSES** (Liberal threshold - lawyer's practical approach):
- Document Name, Agreement Date, Parties, Governing Law, Effective Date, Expiration Date
- Standard: Extract factual information that lawyers need for contract review and analysis
- Approach: Follow legal review patterns - where lawyers actually look for this information
- Priority: Practical legal significance over perfect semantic matching

**CONCEPTUAL CLAUSES** (Strict threshold - require specific legal language):
- Non-Compete, Change of Control, License subtypes (Non-Transferable, Unlimited, etc.)
- Standard: Mark impossible unless explicit, specific legal language creates the provision
- Avoid false positives from related but distinct concepts

**STANDARD CLAUSES** (Balanced threshold - typical legal provisions):
- All other clause types
- Standard: Extract when clear legal obligation or provision exists

## COMMON CONFUSIONS TO AVOID

**CRITICAL - DO NOT CONFUSE THESE CONCEPTS:**

**Exclusivity ≠ Non-Compete**:
- Exclusive rights to sell/distribute ≠ Restriction on competing in business
- "exclusive distributor" ≠ "cannot compete with company"

**License Grant ≠ Specific License Types**:
- Basic "grants right to use" ≠ "non-transferable license"
- Basic license ≠ "unlimited license" or "irrevocable license"

**Assignment Restriction ≠ Change of Control**:
- "cannot assign this agreement" ≠ "triggers on change of ownership"
- General assignment rules ≠ M&A/control change provisions

## PRECISION REQUIREMENTS

**IMPOSSIBLE CLASSIFICATION** - Mark impossible=true when:
- After systematic search, no text directly creates the specific legal obligation
- Only related but distinct concepts found (see confusions above)
- You have thoroughly examined all potentially relevant sections
- **CRITICAL**: When genuinely unsure about relevance → mark impossible=true

**TEXT EXTRACTION STANDARDS**:
- Remove section numbers, titles, and formatting artifacts
- Extract ONLY substantive legal language
- Ensure extracted text directly addresses the clause type
- Provide exact character positions from the document

**QUALITY VALIDATION**:
- Each answer must DIRECTLY create or reference the requested legal provision
- Avoid general contract language that merely mentions keywords
- Apply appropriate threshold based on clause category above

## OUTPUT REQUIREMENTS

You MUST return valid JSON in this exact format:
{{
  "answers": [{{"text": "exact cleaned text from contract", "answer_start": 123}}],
  "is_impossible": false,
  "reasoning": "specific explanation of why this text satisfies the clause requirement"
}}

**COMPLETION CONFIRMATION**: Ensure you have fully decomposed and addressed the user's clause extraction request before providing your final JSON response."""

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
                    
                    # Clean up content for Gemini model's multiple JSON responses
                    if '{' in content:
                        # Find first JSON object
                        json_start = content.find('{')
                        # Find the matching closing brace for the first JSON
                        brace_count = 0
                        json_end = json_start
                        for i, char in enumerate(content[json_start:], json_start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break
                        content = content[json_start:json_end]
                    
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