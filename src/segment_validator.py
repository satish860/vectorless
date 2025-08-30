"""
Segment validator for verifying reasoning quality and searchability.
Validates that the LLM understood the segmentation task correctly.
"""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter


class SegmentValidator:
    """Validates the quality and searchability of document segments."""
    
    def __init__(self):
        # Common legal terms that should appear in legal documents
        self.expected_legal_terms = {
            'contract', 'agreement', 'party', 'parties', 'term', 'termination',
            'breach', 'default', 'notice', 'obligation', 'right', 'liability',
            'payment', 'warranty', 'clause', 'provision', 'govern', 'jurisdiction',
            'confidential', 'intellectual', 'property', 'enforce', 'remedy',
            'damages', 'force majeure', 'arbitration', 'dispute'
        }
        
        # Legal concept categories for validation
        self.legal_concepts = {
            'Contract Formation', 'Parties', 'Terms and Duration', 'Payment Obligations',
            'Termination Rights', 'Warranties', 'Liability', 'Confidentiality',
            'Governing Law', 'Dispute Resolution', 'Force Majeure', 'Intellectual Property'
        }
    
    def validate_sections(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate all sections and return a comprehensive report.
        
        Args:
            sections: List of segmented sections with metadata
            
        Returns:
            Dictionary containing validation results and recommendations
        """
        validation_results = {
            'total_sections': len(sections),
            'validation_passed': True,
            'issues': [],
            'recommendations': [],
            'quality_score': 0.0,
            'searchability_score': 0.0,
            'section_details': []
        }
        
        total_quality_score = 0
        total_searchability_score = 0
        
        for i, section in enumerate(sections):
            section_validation = self._validate_single_section(section, i + 1)
            validation_results['section_details'].append(section_validation)
            
            total_quality_score += section_validation['quality_score']
            total_searchability_score += section_validation['searchability_score']
            
            if section_validation['issues']:
                validation_results['issues'].extend(section_validation['issues'])
                validation_results['validation_passed'] = False
            
            if section_validation['recommendations']:
                validation_results['recommendations'].extend(section_validation['recommendations'])
        
        # Calculate overall scores
        if len(sections) > 0:
            validation_results['quality_score'] = total_quality_score / len(sections)
            validation_results['searchability_score'] = total_searchability_score / len(sections)
        
        # Add overall recommendations
        validation_results['recommendations'].extend(
            self._get_overall_recommendations(sections, validation_results)
        )
        
        return validation_results
    
    def _validate_single_section(self, section: Dict[str, Any], section_num: int) -> Dict[str, Any]:
        """Validate a single section."""
        result = {
            'section_number': section_num,
            'title': section.get('title', ''),
            'quality_score': 0.0,
            'searchability_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Check required fields
        required_fields = ['title', 'summary', 'key_terms', 'legal_concept', 'reasoning']
        missing_fields = [field for field in required_fields if not section.get(field)]
        
        if missing_fields:
            result['issues'].append(f"Section {section_num}: Missing fields: {', '.join(missing_fields)}")
            result['quality_score'] -= 20 * len(missing_fields)
        
        # Validate title quality
        title_score = self._validate_title(section.get('title', ''))
        result['quality_score'] += title_score
        
        # Validate summary quality
        summary_score = self._validate_summary(section.get('summary', ''), section.get('text', ''))
        result['quality_score'] += summary_score
        
        # Validate key terms
        key_terms_score = self._validate_key_terms(
            section.get('key_terms', []), 
            section.get('text', ''),
            section.get('summary', '')
        )
        result['searchability_score'] += key_terms_score
        
        # Validate legal concept
        legal_concept_score = self._validate_legal_concept(section.get('legal_concept', ''))
        result['quality_score'] += legal_concept_score
        
        # Validate reasoning
        reasoning_score = self._validate_reasoning(section.get('reasoning', ''))
        result['quality_score'] += reasoning_score
        
        # Ensure scores are between 0 and 100
        result['quality_score'] = max(0, min(100, result['quality_score'] + 50))  # Base score of 50
        result['searchability_score'] = max(0, min(100, result['searchability_score'] + 50))
        
        return result
    
    def _validate_title(self, title: str) -> float:
        """Validate section title quality."""
        if not title or len(title.strip()) < 3:
            return -20
        
        score = 0
        
        # Check for descriptive length
        if 5 <= len(title.split()) <= 10:
            score += 10
        
        # Check for legal terminology
        title_lower = title.lower()
        legal_term_count = sum(1 for term in self.expected_legal_terms if term in title_lower)
        score += min(10, legal_term_count * 2)
        
        # Penalize very generic titles
        generic_titles = ['section', 'clause', 'paragraph', 'provisions']
        if any(generic in title_lower for generic in generic_titles):
            score -= 5
        
        return score
    
    def _validate_summary(self, summary: str, section_text: str) -> float:
        """Validate summary quality and relevance."""
        if not summary or len(summary.strip()) < 20:
            return -20
        
        score = 0
        
        # Check length (should be 2-3 sentences)
        sentence_count = len(re.split(r'[.!?]+', summary.strip()))
        if 2 <= sentence_count <= 4:
            score += 15
        elif sentence_count == 1:
            score += 5
        elif sentence_count > 6:
            score -= 5
        
        # Check for legal terminology in summary
        summary_lower = summary.lower()
        legal_term_count = sum(1 for term in self.expected_legal_terms if term in summary_lower)
        score += min(15, legal_term_count * 2)
        
        # Check if summary contains key concepts from the actual text
        if section_text:
            text_words = set(re.findall(r'\b\w+\b', section_text.lower()))
            summary_words = set(re.findall(r'\b\w+\b', summary_lower))
            overlap_ratio = len(text_words.intersection(summary_words)) / len(summary_words) if summary_words else 0
            score += overlap_ratio * 10
        
        return score
    
    def _validate_key_terms(self, key_terms: List[str], section_text: str, summary: str) -> float:
        """Validate key terms for searchability."""
        if not key_terms:
            return -30
        
        score = 0
        
        # Check quantity (should have 3-10 terms)
        term_count = len(key_terms)
        if 3 <= term_count <= 10:
            score += 20
        elif term_count < 3:
            score += term_count * 5
        else:
            score += 15  # Too many terms is less problematic than too few
        
        # Check if terms actually appear in the text
        text_lower = section_text.lower() if section_text else ''
        summary_lower = summary.lower() if summary else ''
        combined_content = f"{text_lower} {summary_lower}"
        
        relevant_terms = 0
        for term in key_terms:
            if term.lower() in combined_content:
                relevant_terms += 1
        
        relevance_ratio = relevant_terms / len(key_terms) if key_terms else 0
        score += relevance_ratio * 20
        
        # Bonus for legal terms
        legal_term_count = sum(1 for term in key_terms if term.lower() in self.expected_legal_terms)
        score += min(10, legal_term_count * 2)
        
        return score
    
    def _validate_legal_concept(self, legal_concept: str) -> float:
        """Validate legal concept categorization."""
        if not legal_concept:
            return -15
        
        score = 0
        
        # Check if it's a recognized legal concept
        concept_lower = legal_concept.lower()
        if any(known.lower() in concept_lower or concept_lower in known.lower() 
               for known in self.legal_concepts):
            score += 15
        else:
            # Check for legal terminology
            legal_term_count = sum(1 for term in self.expected_legal_terms if term in concept_lower)
            score += min(10, legal_term_count * 3)
        
        return score
    
    def _validate_reasoning(self, reasoning: str) -> float:
        """Validate the quality of CoT reasoning."""
        if not reasoning or len(reasoning.strip()) < 20:
            return -10
        
        score = 0
        
        # Check for reasoning indicators
        reasoning_indicators = [
            'because', 'since', 'therefore', 'thus', 'establishes', 'covers',
            'defines', 'specifies', 'includes', 'requires', 'provisions'
        ]
        
        reasoning_lower = reasoning.lower()
        indicator_count = sum(1 for indicator in reasoning_indicators if indicator in reasoning_lower)
        score += min(10, indicator_count * 2)
        
        # Check for legal terminology in reasoning
        legal_term_count = sum(1 for term in self.expected_legal_terms if term in reasoning_lower)
        score += min(10, legal_term_count * 1)
        
        return score
    
    def _get_overall_recommendations(self, sections: List[Dict[str, Any]], 
                                   validation_results: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on validation results."""
        recommendations = []
        
        if validation_results['quality_score'] < 70:
            recommendations.append("Overall quality is below threshold. Consider improving summaries and reasoning.")
        
        if validation_results['searchability_score'] < 70:
            recommendations.append("Searchability could be improved. Add more relevant key terms.")
        
        # Check for section coverage gaps
        covered_concepts = set()
        for section in sections:
            if section.get('legal_concept'):
                covered_concepts.add(section['legal_concept'].lower())
        
        if len(covered_concepts) < len(sections) * 0.7:
            recommendations.append("Some sections may have overlapping legal concepts. Consider more distinct categorization.")
        
        return recommendations
    
    def print_validation_report(self, validation_results: Dict[str, Any]) -> None:
        """Print a formatted validation report."""
        print("\n" + "="*100)
        print("SEGMENT VALIDATION REPORT")
        print("="*100)
        
        print(f"Overall Quality Score: {validation_results['quality_score']:.1f}/100")
        print(f"Overall Searchability Score: {validation_results['searchability_score']:.1f}/100")
        print(f"Validation Status: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        
        if validation_results['issues']:
            print(f"\nISSUES FOUND ({len(validation_results['issues'])}):")
            for issue in validation_results['issues'][:10]:  # Limit to first 10
                print(f"  - {issue}")
            if len(validation_results['issues']) > 10:
                print(f"  ... and {len(validation_results['issues']) - 10} more issues")
        
        if validation_results['recommendations']:
            print(f"\nRECOMMENDAITONS ({len(validation_results['recommendations'])}):")
            for rec in validation_results['recommendations']:
                print(f"  - {rec}")
        
        print("\nPER-SECTION SCORES:")
        print("-" * 100)
        for detail in validation_results['section_details'][:5]:  # Show first 5 sections
            print(f"Section {detail['section_number']}: {detail['title']}")
            print(f"  Quality: {detail['quality_score']:.1f}/100, Searchability: {detail['searchability_score']:.1f}/100")
            if detail['issues']:
                print(f"  Issues: {'; '.join(detail['issues'])}")
        
        if len(validation_results['section_details']) > 5:
            print(f"  ... and {len(validation_results['section_details']) - 5} more sections")
        
        print("="*100)


def validate_segmentation_file(filepath: str) -> Dict[str, Any]:
    """
    Load and validate a segmentation file.
    
    Args:
        filepath: Path to the segmentation JSON file
        
    Returns:
        Validation results dictionary
    """
    import json
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sections = data.get('sections', [])
        validator = SegmentValidator()
        results = validator.validate_sections(sections)
        
        # Add file information
        results['source_file'] = filepath
        results['contract_title'] = data.get('contract_title', 'Unknown')
        
        return results
        
    except Exception as e:
        return {
            'validation_passed': False,
            'issues': [f"Failed to validate file {filepath}: {str(e)}"],
            'quality_score': 0.0,
            'searchability_score': 0.0
        }