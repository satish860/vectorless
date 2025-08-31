#!/usr/bin/env python3
"""
Complete contract processing pipeline for CUAD dataset.
Handles contract loading, segmentation (with caching), extraction, and evaluation.
"""

import json
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from cuad_tool_extractor import CUADToolExtractor
    from segmentation import segment_document, save_segmentation_to_json, load_segmentation_from_json
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure the src/ directory contains the required modules")
    sys.exit(1)


class ContractProcessor:
    """Complete contract processing pipeline for CUAD dataset."""
    
    def __init__(self, cuad_dataset_path: str = "data/CUADv1.json"):
        """
        Initialize the contract processor.
        
        Args:
            cuad_dataset_path: Path to the CUAD dataset JSON file
        """
        # Adjust path to be relative to project root
        if not os.path.isabs(cuad_dataset_path):
            cuad_dataset_path = str(Path(__file__).parent.parent / cuad_dataset_path)
        self.cuad_dataset_path = cuad_dataset_path
        self.cuad_data = None
        self.extractor = None
        self._load_cuad_dataset()
        
        # Ensure output directories exist relative to project root
        project_root = Path(__file__).parent.parent
        os.makedirs(project_root / "output" / "results", exist_ok=True)
        os.makedirs(project_root / "output" / "segmentation_results", exist_ok=True)
    
    def _load_cuad_dataset(self):
        """Load the CUAD dataset."""
        try:
            with open(self.cuad_dataset_path, 'r', encoding='utf-8') as f:
                self.cuad_data = json.load(f)
            print(f"Loaded CUAD dataset with {len(self.cuad_data.get('data', []))} contracts")
        except FileNotFoundError:
            raise FileNotFoundError(f"CUAD dataset not found at {self.cuad_dataset_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {self.cuad_dataset_path}")
    
    def get_contract_by_index(self, index: int) -> Dict[str, Any]:
        """
        Get contract data by index.
        
        Args:
            index: Zero-based index in the dataset
            
        Returns:
            Dictionary containing contract data
        """
        if not self.cuad_data or index < 0 or index >= len(self.cuad_data['data']):
            raise ValueError(f"Invalid contract index {index}")
        
        return self.cuad_data['data'][index]
    
    def get_contract_by_title(self, title: str) -> Tuple[Dict[str, Any], int]:
        """
        Get contract data by title (fuzzy matching).
        
        Args:
            title: Contract title to search for
            
        Returns:
            Tuple of (contract_data, index)
        """
        if not self.cuad_data:
            raise ValueError("CUAD dataset not loaded")
        
        title_lower = title.lower()
        
        for i, contract in enumerate(self.cuad_data['data']):
            contract_title = contract.get('title', '').lower()
            if title_lower in contract_title or contract_title in title_lower:
                return contract, i
        
        raise ValueError(f"No contract found with title containing: {title}")
    
    def list_contracts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List available contracts with their indices and titles.
        
        Args:
            limit: Maximum number of contracts to show
            
        Returns:
            List of contract info dictionaries
        """
        if not self.cuad_data:
            return []
        
        contracts_info = []
        for i, contract in enumerate(self.cuad_data['data'][:limit]):
            contracts_info.append({
                'index': i,
                'title': contract.get('title', f'Contract {i}'),
                'paragraphs': len(contract.get('paragraphs', [])),
                'questions': sum(len(p.get('qas', [])) for p in contract.get('paragraphs', []))
            })
        
        return contracts_info
    
    def _get_cache_filepath(self, contract_title: str) -> str:
        """Get the filepath for cached segmentation results."""
        # Clean filename - remove special characters
        clean_title = "".join(c for c in contract_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title.replace(' ', '_')[:50]  # Limit length
        project_root = Path(__file__).parent.parent
        return str(project_root / "output" / "segmentation_results" / f"{clean_title}_cached.json")
    
    def get_segmentation(self, contract_text: str, contract_title: str, force_new: bool = False) -> Dict[str, Any]:
        """
        Get contract segmentation, reusing cache if available.
        
        Args:
            contract_text: The contract text to segment
            contract_title: Title of the contract
            force_new: Force new segmentation even if cache exists
            
        Returns:
            Dictionary containing segmentation data
        """
        cache_filepath = self._get_cache_filepath(contract_title)
        
        # Try to load from cache first
        if not force_new and os.path.exists(cache_filepath):
            try:
                print(f"Loading cached segmentation from: {cache_filepath}")
                cached_data = load_segmentation_from_json(cache_filepath)
                return {
                    "contract_title": cached_data["contract_title"],
                    "sections": cached_data["sections"],
                    "total_sections": cached_data["total_sections"],
                    "cached": True,
                    "cache_filepath": cache_filepath
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Proceeding with new segmentation...")
        
        # Perform new segmentation
        print(f"Creating new segmentation for: {contract_title}")
        sections = segment_document(contract_text, save_results=False)
        
        # Save with proper contract title and as cache file
        try:
            save_segmentation_to_json(sections, contract_title)
            
            # Also save as cache file
            result_data = {
                "contract_title": contract_title,
                "timestamp": datetime.now().isoformat(),
                "total_sections": len(sections),
                "sections": sections
            }
            with open(cache_filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            print(f"Cached results saved to: {cache_filepath}")
        except Exception as e:
            print(f"Warning: Failed to save results: {e}")
        
        return {
            "contract_title": contract_title,
            "sections": sections,
            "total_sections": len(sections),
            "cached": False,
            "cache_filepath": cache_filepath
        }
    
    def extract_questions_from_contract(self, contract_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all questions from a contract.
        
        Args:
            contract_data: Contract data from CUAD dataset
            
        Returns:
            List of question dictionaries
        """
        questions = []
        for paragraph in contract_data.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                questions.append({
                    "id": qa.get("id", ""),
                    "question": qa.get("question", ""),
                    "ground_truth_answers": qa.get("answers", []),
                    "is_impossible": qa.get("is_impossible", False),
                    "clause_type": qa.get("id", "").split("__")[-1] if "__" in qa.get("id", "") else "Unknown"
                })
        return questions
    
    def get_contract_text(self, contract_data: Dict[str, Any]) -> str:
        """
        Extract the contract text from CUAD contract data.
        
        Args:
            contract_data: Contract data from CUAD dataset
            
        Returns:
            The contract text
        """
        if not contract_data.get('paragraphs') or not contract_data['paragraphs'][0].get('context'):
            raise ValueError("No contract context found in contract data")
        
        return contract_data['paragraphs'][0]['context']
    
    def calculate_text_overlap(self, predicted: str, ground_truth: str) -> float:
        """Calculate text overlap between predicted and ground truth answers."""
        if not predicted or not ground_truth:
            return 0.0
        
        # Enhanced normalization for better exact matching
        def normalize_text(text: str) -> str:
            """Enhanced text normalization to handle common formatting differences."""
            # Basic normalization
            normalized = text.lower().strip()
            
            # Remove extra whitespace and normalize quotes
            normalized = ' '.join(normalized.split())  # Collapse multiple spaces
            normalized = normalized.replace('"', '"').replace('"', '"')  # Normalize smart quotes
            normalized = normalized.replace(''', "'").replace(''', "'")  # Normalize smart apostrophes
            
            # Remove common formatting artifacts
            normalized = normalized.replace('  ', ' ')  # Double spaces
            normalized = normalized.strip('.,;:')  # Leading/trailing punctuation
            
            return normalized
        
        pred_normalized = normalize_text(predicted)
        gt_normalized = normalize_text(ground_truth)
        
        # Exact match after enhanced normalization
        if pred_normalized == gt_normalized:
            return 1.0
        
        # Check if one contains the other (allow slight variations)
        if pred_normalized in gt_normalized or gt_normalized in pred_normalized:
            return 0.95  # Higher score for containment after normalization
        
        # Check for near-exact matches with minor differences
        # Remove all punctuation and compare
        def remove_punctuation(text: str) -> str:
            import string
            return ''.join(c for c in text if c not in string.punctuation)
        
        pred_no_punct = remove_punctuation(pred_normalized)
        gt_no_punct = remove_punctuation(gt_normalized)
        
        if pred_no_punct == gt_no_punct:
            return 0.92  # Very high score for punctuation-only differences
        
        if pred_no_punct in gt_no_punct or gt_no_punct in pred_no_punct:
            return 0.85  # High score for containment without punctuation
        
        # Word-level overlap
        pred_words = set(pred_normalized.split())
        gt_words = set(gt_normalized.split())
        
        if not gt_words:
            return 0.0
        
        overlap = len(pred_words.intersection(gt_words))
        return overlap / len(gt_words)
    
    def evaluate_answer(self, predicted_result: Dict, ground_truth: Dict) -> Dict[str, Any]:
        """Evaluate a single answer against ground truth."""
        evaluation = {
            "question_id": ground_truth["id"],
            "clause_type": ground_truth["clause_type"],
            "predicted_impossible": predicted_result.get("is_impossible", False),
            "ground_truth_impossible": ground_truth["is_impossible"],
            "impossible_correct": False,
            "text_matches": [],
            "best_overlap": 0.0,
            "exact_matches": 0,
            "partial_matches": 0,
            "total_ground_truth": len(ground_truth["ground_truth_answers"])
        }
        
        # Check if impossible prediction is correct
        evaluation["impossible_correct"] = (
            evaluation["predicted_impossible"] == evaluation["ground_truth_impossible"]
        )
        
        # If both agree it's impossible, that's a perfect match
        if evaluation["predicted_impossible"] and evaluation["ground_truth_impossible"]:
            evaluation["best_overlap"] = 1.0
            return evaluation
        
        # If ground truth is impossible but predicted answers, that's wrong
        if evaluation["ground_truth_impossible"] and not evaluation["predicted_impossible"]:
            return evaluation
        
        # Compare text answers
        predicted_texts = [ans.get("text", "") for ans in predicted_result.get("answers", [])]
        ground_truth_texts = [ans.get("text", "") for ans in ground_truth["ground_truth_answers"]]
        
        if not ground_truth_texts:
            # No ground truth answers, so any prediction is wrong unless marked impossible
            return evaluation
        
        # Find best matches
        for pred_text in predicted_texts:
            best_match_score = 0.0
            best_gt_text = ""
            
            for gt_text in ground_truth_texts:
                overlap = self.calculate_text_overlap(pred_text, gt_text)
                if overlap > best_match_score:
                    best_match_score = overlap
                    best_gt_text = gt_text
            
            if best_match_score >= 0.9:
                evaluation["exact_matches"] += 1
            elif best_match_score >= 0.5:
                evaluation["partial_matches"] += 1
            
            evaluation["text_matches"].append({
                "predicted": pred_text[:100] + "..." if len(pred_text) > 100 else pred_text,
                "best_ground_truth": best_gt_text[:100] + "..." if len(best_gt_text) > 100 else best_gt_text,
                "overlap_score": best_match_score
            })
            
            evaluation["best_overlap"] = max(evaluation["best_overlap"], best_match_score)
        
        return evaluation
    
    def process_single_question(self, args):
        """Process a single question - designed for parallel execution."""
        question, question_num, total_questions = args
        
        print(f"[{question_num}/{total_questions}] Processing: {question['clause_type']}")
        
        try:
            # Run extraction
            predicted_result = self.extractor.extract_clause(
                question["question"], 
                question["clause_type"]
            )
            
            # Evaluate
            evaluation = self.evaluate_answer(predicted_result, question)
            evaluation["question_number"] = question_num
            
            print(f"[{question_num}/{total_questions}] Completed: {question['clause_type']} - Score: {evaluation['best_overlap']:.2f}")
            return evaluation
            
        except Exception as e:
            print(f"[{question_num}/{total_questions}] ERROR: {question['clause_type']} - {e}")
            return {
                "question_id": question["id"],
                "clause_type": question["clause_type"],
                "question_number": question_num,
                "error": str(e),
                "best_overlap": 0.0,
                "impossible_correct": False,
                "exact_matches": 0,
                "partial_matches": 0,
                "total_ground_truth": len(question["ground_truth_answers"])
            }
    
    def process_contract(self, contract_index: int = None, contract_title: str = None, 
                        force_segment: bool = False, evaluate: bool = True,
                        max_workers: int = 4) -> Dict[str, Any]:
        """
        Process a single contract through the complete pipeline.
        
        Args:
            contract_index: Index of contract in dataset
            contract_title: Title of contract to find
            force_segment: Force new segmentation even if cache exists
            evaluate: Whether to evaluate against ground truth
            max_workers: Number of parallel workers for question processing
            
        Returns:
            Dictionary containing processing results
        """
        # Get contract data
        if contract_index is not None:
            contract_data = self.get_contract_by_index(contract_index)
            actual_index = contract_index
        elif contract_title is not None:
            contract_data, actual_index = self.get_contract_by_title(contract_title)
        else:
            raise ValueError("Must specify either contract_index or contract_title")
        
        title = contract_data.get('title', f'Contract {actual_index}')
        print(f"\n{'='*80}")
        print(f"PROCESSING CONTRACT: {title}")
        print(f"Index: {actual_index}")
        print(f"{'='*80}")
        
        # Get contract text and questions
        contract_text = self.get_contract_text(contract_data)
        questions = self.extract_questions_from_contract(contract_data)
        
        print(f"Contract text length: {len(contract_text):,} characters")
        print(f"Questions to process: {len(questions)}")
        
        # Get segmentation (with caching)
        print(f"\nStep 1: Document Segmentation")
        segmentation_result = self.get_segmentation(contract_text, title, force_segment)
        print(f"Segmentation: {segmentation_result['total_sections']} sections " +
              f"({'cached' if segmentation_result['cached'] else 'newly created'})")
        
        # Initialize extractor and load segmentation
        print(f"\nStep 2: Initialize Extractor")
        self.extractor = CUADToolExtractor()
        self.extractor.load_segmentation(segmentation_result['cache_filepath'])
        
        # Process questions
        print(f"\nStep 3: Extract Clauses")
        question_args = [(question, i, len(questions)) for i, question in enumerate(questions, 1)]
        
        results = []
        completed_count = 0
        total_questions = len(questions)
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(max_workers, total_questions)
        print(f"Processing {total_questions} questions with {max_workers} parallel workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all questions for processing
            future_to_question = {
                executor.submit(self.process_single_question, args): args[1] 
                for args in question_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_question):
                question_num = future_to_question[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                except Exception as e:
                    print(f"Failed question {question_num}: {e}")
                    results.append({
                        "question_id": f"question_{question_num}",
                        "clause_type": "unknown",
                        "question_number": question_num,
                        "error": str(e),
                        "best_overlap": 0.0
                    })
                    completed_count += 1
        
        # Sort results by question number to maintain order
        results.sort(key=lambda x: x.get("question_number", 0))
        
        # Calculate metrics
        print(f"\nStep 4: Calculate Metrics")
        total_score = sum(r.get("best_overlap", 0) for r in results)
        
        metrics = {}
        if results:
            avg_accuracy = total_score / len(results)
            impossible_correct = sum(1 for r in results if r.get("impossible_correct", False))
            total_exact_matches = sum(r.get("exact_matches", 0) for r in results)
            total_partial_matches = sum(r.get("partial_matches", 0) for r in results)
            total_ground_truth_answers = sum(r.get("total_ground_truth", 0) for r in results)
            
            metrics = {
                "total_questions": len(results),
                "avg_accuracy": avg_accuracy,
                "impossible_classification_accuracy": impossible_correct / len(results),
                "total_exact_matches": total_exact_matches,
                "total_partial_matches": total_partial_matches,
                "total_ground_truth_answers": total_ground_truth_answers,
                "estimated_recall": (total_exact_matches + total_partial_matches * 0.5) / total_ground_truth_answers if total_ground_truth_answers > 0 else 0
            }
            
            print(f"Overall Accuracy: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
            print(f"Impossible Classification: {impossible_correct}/{len(results)} ({impossible_correct/len(results)*100:.1f}%)")
            print(f"Exact Matches: {total_exact_matches}")
            print(f"Partial Matches: {total_partial_matches}")
            print(f"Ground Truth Answers: {total_ground_truth_answers}")
            if total_ground_truth_answers > 0:
                print(f"Estimated Recall: {metrics['estimated_recall']:.3f} ({metrics['estimated_recall']*100:.1f}%)")
        
        # Save results
        result_data = {
            "contract_info": {
                "index": actual_index,
                "title": title,
                "processing_timestamp": datetime.now().isoformat(),
                "segmentation_cached": segmentation_result['cached'],
                "total_sections": segmentation_result['total_sections']
            },
            "summary": metrics,
            "detailed_results": results
        }
        
        # Save to file
        project_root = Path(__file__).parent.parent
        results_filename = str(project_root / "output" / "results" / f"{title.replace(' ', '_')[:50]}_extraction_results.json")
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {results_filename}")
        
        return result_data
    
    def batch_process_contracts(self, start_index: int = 0, count: int = 5, 
                               force_segment: bool = False, max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Process multiple contracts in batch.
        
        Args:
            start_index: Starting index in dataset
            count: Number of contracts to process
            force_segment: Force new segmentation for all contracts
            max_workers: Number of parallel workers per contract
            
        Returns:
            List of processing results
        """
        if not self.cuad_data:
            raise ValueError("CUAD dataset not loaded")
        
        max_index = len(self.cuad_data['data'])
        end_index = min(start_index + count, max_index)
        
        print(f"BATCH PROCESSING: Contracts {start_index} to {end_index-1} ({end_index-start_index} total)")
        
        batch_results = []
        for i in range(start_index, end_index):
            try:
                print(f"\n{'#'*20} BATCH ITEM {i-start_index+1}/{end_index-start_index} {'#'*20}")
                result = self.process_contract(contract_index=i, force_segment=force_segment, max_workers=max_workers)
                batch_results.append(result)
            except Exception as e:
                print(f"Failed to process contract {i}: {e}")
                batch_results.append({
                    "contract_info": {"index": i, "error": str(e)},
                    "summary": {},
                    "detailed_results": []
                })
        
        # Save batch summary
        batch_summary = {
            "batch_info": {
                "start_index": start_index,
                "end_index": end_index-1,
                "total_processed": len(batch_results),
                "processing_timestamp": datetime.now().isoformat()
            },
            "results": batch_results
        }
        
        project_root = Path(__file__).parent.parent
        batch_filename = str(project_root / "output" / "results" / f"batch_{start_index}_{end_index-1}_results.json")
        with open(batch_filename, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nBatch results saved to: {batch_filename}")
        
        return batch_results


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(description="Process CUAD contracts through segmentation and extraction pipeline")
    
    # Contract selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--contract-index", type=int, help="Process contract by index")
    group.add_argument("--contract-title", type=str, help="Process contract by title (fuzzy search)")
    group.add_argument("--batch", type=int, help="Batch process N contracts starting from index 0")
    group.add_argument("--batch-range", nargs=2, type=int, metavar=("START", "COUNT"), 
                      help="Batch process COUNT contracts starting from START index")
    group.add_argument("--list", action="store_true", help="List available contracts")
    
    # Processing options
    parser.add_argument("--force-segment", action="store_true", 
                       help="Force new segmentation even if cache exists")
    parser.add_argument("--no-evaluate", action="store_true", 
                       help="Skip evaluation against ground truth")
    parser.add_argument("--max-workers", type=int, default=4, 
                       help="Maximum parallel workers for question processing (default: 4)")
    parser.add_argument("--dataset", type=str, default="data/CUADv1.json", 
                       help="Path to CUAD dataset (default: data/CUADv1.json)")
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = ContractProcessor(args.dataset)
        
        if args.list:
            # List contracts
            contracts = processor.list_contracts(20)  # Show first 20
            print("Available Contracts:")
            print("-" * 80)
            for contract in contracts:
                print(f"{contract['index']:3d}: {contract['title']}")
                print(f"     Paragraphs: {contract['paragraphs']}, Questions: {contract['questions']}")
            return
        
        # Process contracts
        if args.batch:
            processor.batch_process_contracts(
                start_index=0, 
                count=args.batch, 
                force_segment=args.force_segment,
                max_workers=args.max_workers
            )
        elif args.batch_range:
            start, count = args.batch_range
            processor.batch_process_contracts(
                start_index=start, 
                count=count, 
                force_segment=args.force_segment,
                max_workers=args.max_workers
            )
        else:
            # Single contract processing
            processor.process_contract(
                contract_index=args.contract_index,
                contract_title=args.contract_title,
                force_segment=args.force_segment,
                evaluate=not args.no_evaluate,
                max_workers=args.max_workers
            )
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)