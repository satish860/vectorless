#!/usr/bin/env python3
"""
Run all 41 questions from sample_cuad.json through the CUAD tool extractor
and evaluate accuracy against ground truth answers.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src directory to path to import cuad_tool_extractor
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from cuad_tool_extractor import CUADToolExtractor
except ImportError as e:
    print(f"Error importing CUADToolExtractor: {e}")
    print("Make sure the src/cuad_tool_extractor.py file exists and dependencies are installed")
    sys.exit(1)


def load_sample_questions(limit: int = 41) -> List[Dict[str, Any]]:
    """Load the first N questions from sample CUAD dataset."""
    sample_path = Path(__file__).parent.parent / "sample_dataset/sample_cuad.json"
    
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample dataset not found at {sample_path}")
    
    with open(sample_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    for item in data.get("data", []):
        for paragraph in item.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                questions.append({
                    "id": qa.get("id", ""),
                    "question": qa.get("question", ""),
                    "ground_truth_answers": qa.get("answers", []),
                    "is_impossible": qa.get("is_impossible", False),
                    "clause_type": qa.get("id", "").split("__")[-1] if "__" in qa.get("id", "") else "Unknown"
                })
                
                if len(questions) >= limit:
                    return questions
    
    return questions


def calculate_text_overlap(predicted: str, ground_truth: str) -> float:
    """Calculate text overlap between predicted and ground truth answers."""
    if not predicted or not ground_truth:
        return 0.0
    
    # Normalize text (lowercase, strip whitespace)
    pred_normalized = predicted.lower().strip()
    gt_normalized = ground_truth.lower().strip()
    
    # Exact match
    if pred_normalized == gt_normalized:
        return 1.0
    
    # Check if one contains the other
    if pred_normalized in gt_normalized or gt_normalized in pred_normalized:
        return 0.8
    
    # Word-level overlap
    pred_words = set(pred_normalized.split())
    gt_words = set(gt_normalized.split())
    
    if not gt_words:
        return 0.0
    
    overlap = len(pred_words.intersection(gt_words))
    return overlap / len(gt_words)


def evaluate_answer(predicted_result: Dict, ground_truth: Dict) -> Dict[str, Any]:
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
            overlap = calculate_text_overlap(pred_text, gt_text)
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


def process_single_question(args):
    """Process a single question - designed for parallel execution."""
    question, extractor, question_num = args
    
    print(f"Processing question {question_num}: {question['clause_type']}")
    
    try:
        # Run extraction
        predicted_result = extractor.extract_clause(
            question["question"], 
            question["clause_type"]
        )
        
        # Evaluate
        evaluation = evaluate_answer(predicted_result, question)
        evaluation["question_number"] = question_num
        
        print(f"[OK] Completed question {question_num}: {question['clause_type']} - Score: {evaluation['best_overlap']:.2f}")
        return evaluation
        
    except Exception as e:
        print(f"[ERROR] Error processing question {question_num}: {e}")
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


def main():
    """Main function to run the evaluation."""
    print("=" * 80)
    print("CUAD Tool Extractor - All 41 Questions Evaluation")
    print("=" * 80)
    
    try:
        # Load questions
        print("Loading all 41 questions from sample dataset...")
        questions = load_sample_questions(41)
        print(f"Loaded {len(questions)} questions")
        
        # Initialize extractor
        print("Initializing CUAD Tool Extractor...")
        extractor = CUADToolExtractor()
        
        # Load segmentation (you may need to adjust this path)
        segmentation_file = Path(__file__).parent.parent / "output/segmentation_results/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR_AGREEMEN_cached.json"
        if not segmentation_file.exists():
            print(f"Warning: Segmentation file not found at {segmentation_file}")
            print("You may need to run document segmentation first")
            return
        
        extractor.load_segmentation(str(segmentation_file))
        
        print("\n" + "=" * 80)
        print("RUNNING EXTRACTIONS IN PARALLEL")
        print("=" * 80)
        
        # Prepare arguments for parallel processing
        question_args = [(question, extractor, i) for i, question in enumerate(questions, 1)]
        
        # Run extractions in parallel
        results = []
        completed_count = 0
        total_questions = len(questions)
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(8, total_questions)  # Limit to 8 concurrent threads
        print(f"Processing {total_questions} questions with {max_workers} parallel workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all questions for processing
            future_to_question = {
                executor.submit(process_single_question, args): args[2] 
                for args in question_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_question):
                question_num = future_to_question[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    print(f"[{completed_count}/{total_questions}] Completed question {question_num}")
                    
                except Exception as e:
                    print(f"[{completed_count+1}/{total_questions}] Failed question {question_num}: {e}")
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
        total_score = sum(r.get("best_overlap", 0) for r in results)
        
        # Calculate overall metrics
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        if results:
            avg_accuracy = total_score / len(results)
            impossible_correct = sum(1 for r in results if r.get("impossible_correct", False))
            total_exact_matches = sum(r.get("exact_matches", 0) for r in results)
            total_partial_matches = sum(r.get("partial_matches", 0) for r in results)
            total_ground_truth_answers = sum(r.get("total_ground_truth", 0) for r in results)
            
            print(f"Overall Accuracy Score: {avg_accuracy:.2f} ({avg_accuracy*100:.1f}%)")
            print(f"Impossible Classification Accuracy: {impossible_correct}/{len(results)} ({impossible_correct/len(results)*100:.1f}%)")
            print(f"Total Exact Matches: {total_exact_matches}")
            print(f"Total Partial Matches: {total_partial_matches}")
            print(f"Total Ground Truth Answers: {total_ground_truth_answers}")
            
            if total_ground_truth_answers > 0:
                recall = (total_exact_matches + total_partial_matches * 0.5) / total_ground_truth_answers
                print(f"Estimated Recall: {recall:.2f} ({recall*100:.1f}%)")
            
            print("\nDetailed Results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['clause_type']}: {result.get('best_overlap', 0):.2f}")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
        
        # Save detailed results
        output_file = Path(__file__).parent.parent / "output" / "all_41_questions_evaluation.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_questions": len(results),
                    "avg_accuracy": avg_accuracy if results else 0,
                    "impossible_classification_accuracy": impossible_correct / len(results) if results else 0,
                    "total_exact_matches": total_exact_matches,
                    "total_partial_matches": total_partial_matches,
                    "total_ground_truth_answers": total_ground_truth_answers
                },
                "detailed_results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)