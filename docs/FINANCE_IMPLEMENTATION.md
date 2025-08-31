# Finance Document Processing Implementation

## Project Structure for Finance Documents

```
vectorless/
├── src/
│   ├── core/                    # Reusable core components
│   │   ├── base_processor.py   # Abstract base class
│   │   ├── segmentation.py     # Document segmentation
│   │   └── evaluation.py       # Evaluation framework
│   ├── legal/                   # Legal domain implementation
│   │   └── cuad_tool_extractor.py
│   └── finance/                 # Finance domain implementation
│       ├── finance_processor.py # Finance-specific processor
│       ├── finance_extractor.py # Finance Q&A extractor
│       └── finance_questions.py # Finance question templates
├── scripts/
│   ├── legal/                   # Legal processing scripts
│   │   ├── process_contract.py
│   │   └── run_all_41_questions.py
│   └── finance/                 # Finance processing scripts
│       ├── process_10k.py
│       ├── process_earnings.py
│       └── run_finance_analysis.py
├── data/
│   ├── legal/                   # Legal datasets
│   │   └── CUADv1.json
│   └── finance/                 # Finance datasets
│       ├── 10k_reports/
│       ├── 10q_reports/
│       └── earnings_transcripts/
├── output/
│   ├── legal/                   # Legal results
│   └── finance/                 # Finance results
└── docs/
    ├── legal_workflow.md
    └── finance_workflow.md
```

## Finance-Specific Components

### 1. Finance Document Types & Questions

#### 10-K Annual Reports
```python
TEN_K_QUESTIONS = {
    "financial_performance": [
        "What was the total revenue for the fiscal year?",
        "What was the net income for the fiscal year?", 
        "What was the operating cash flow?",
        "What was the total debt at year end?"
    ],
    "risk_factors": [
        "What are the primary business risks?",
        "Are there any regulatory risks mentioned?",
        "What operational risks does the company face?",
        "Are there any cybersecurity risks disclosed?"
    ],
    "business_overview": [
        "What is the company's primary business?",
        "Who are the main competitors?",
        "What geographic markets does the company serve?",
        "What are the main revenue streams?"
    ],
    "governance": [
        "Are there any material weaknesses in internal controls?",
        "What are the executive compensation details?",
        "Are there any related party transactions?",
        "What audit firm was used?"
    ]
}
```

#### 10-Q Quarterly Reports  
```python
TEN_Q_QUESTIONS = {
    "quarterly_performance": [
        "What was the revenue for this quarter?",
        "What was the net income for this quarter?",
        "How does this quarter compare to the same quarter last year?",
        "What was the diluted EPS for this quarter?"
    ],
    "liquidity": [
        "What is the current cash position?",
        "What are the current debt levels?",
        "Are there any liquidity concerns mentioned?",
        "What credit facilities are available?"
    ]
}
```

#### Earnings Call Transcripts
```python
EARNINGS_QUESTIONS = {
    "guidance": [
        "What guidance was provided for next quarter?",
        "What guidance was provided for the full year?",
        "Were there any guidance revisions?",
        "What are the key assumptions behind the guidance?"
    ],
    "management_commentary": [
        "What did management say about market conditions?",
        "What operational highlights were mentioned?",
        "Are there any new strategic initiatives?",
        "What challenges were discussed?"
    ]
}
```

### 2. Finance Document Segmentation

```python
def segment_financial_document(doc_text: str, doc_type: str) -> List[Dict]:
    """
    Segment financial documents based on document type.
    
    Args:
        doc_text: The document text
        doc_type: Type of document ('10-K', '10-Q', 'earnings')
        
    Returns:
        List of document segments with metadata
    """
    if doc_type == '10-K':
        return segment_10k_document(doc_text)
    elif doc_type == '10-Q':
        return segment_10q_document(doc_text)
    elif doc_type == 'earnings':
        return segment_earnings_transcript(doc_text)
    else:
        return generic_segment_document(doc_text)

def segment_10k_document(doc_text: str) -> List[Dict]:
    """Segment 10-K into standard sections."""
    sections = []
    
    # Key 10-K sections to identify
    section_patterns = {
        "Item 1": "Business",
        "Item 1A": "Risk Factors", 
        "Item 2": "Properties",
        "Item 3": "Legal Proceedings",
        "Item 7": "Management's Discussion and Analysis",
        "Item 8": "Financial Statements",
        "Item 9A": "Controls and Procedures"
    }
    
    # Implementation would parse based on these patterns
    return sections
```

### 3. Finance-Specific Evaluation

```python
def evaluate_financial_answer(predicted: str, ground_truth: str, 
                            answer_type: str = "text") -> float:
    """
    Evaluate financial answers with domain-specific logic.
    
    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        answer_type: Type of answer ("numerical", "text", "currency")
    """
    if answer_type == "numerical":
        return evaluate_numerical_answer(predicted, ground_truth)
    elif answer_type == "currency":
        return evaluate_currency_answer(predicted, ground_truth)
    elif answer_type == "percentage":
        return evaluate_percentage_answer(predicted, ground_truth)
    else:
        return calculate_text_overlap(predicted, ground_truth)

def evaluate_numerical_answer(predicted: str, ground_truth: str) -> float:
    """Evaluate numerical answers with tolerance for formatting."""
    # Extract numbers, handle millions/billions/thousands
    pred_num = parse_financial_number(predicted)
    truth_num = parse_financial_number(ground_truth)
    
    if pred_num is None or truth_num is None:
        return 0.0
    
    # Allow 5% tolerance for rounding differences
    if abs(pred_num - truth_num) / truth_num <= 0.05:
        return 1.0
    elif abs(pred_num - truth_num) / truth_num <= 0.1:
        return 0.8
    else:
        return 0.0
```

### 4. Sample Finance Processor

```python
class FinanceProcessor(BaseProcessor):
    """Finance document processor extending base functionality."""
    
    def __init__(self, dataset_path: str = "data/finance/"):
        super().__init__(dataset_path)
        self.finance_extractor = FinanceExtractor()
    
    def process_10k_report(self, filing_path: str, 
                          questions: List[str] = None) -> Dict:
        """Process a 10-K filing through the pipeline."""
        
        # Load document
        doc_text = self.load_document(filing_path)
        
        # Segment document
        segments = segment_financial_document(doc_text, '10-K')
        
        # Use default questions if none provided
        if questions is None:
            questions = self.get_default_10k_questions()
        
        # Process questions in parallel
        results = self.process_questions_parallel(
            segments, questions, max_workers=8
        )
        
        return {
            "document_path": filing_path,
            "segments_count": len(segments),
            "questions_processed": len(questions),
            "results": results,
            "summary_metrics": self.calculate_finance_metrics(results)
        }
    
    def get_default_10k_questions(self) -> List[Dict]:
        """Get standard 10-K analysis questions."""
        return [
            {"question": q, "type": "text", "category": cat}
            for cat, questions in TEN_K_QUESTIONS.items()
            for q in questions
        ]
```

## Implementation Approach

### Phase 1: Core Refactoring
1. **Extract base classes** from existing legal implementation
2. **Move domain-specific code** to `src/legal/`
3. **Create abstract interfaces** for processors and extractors

### Phase 2: Finance Implementation
1. **Implement finance processor** extending base classes
2. **Create finance question templates** for different document types
3. **Build finance-specific evaluation** with numerical accuracy

### Phase 3: Scripts & Integration
1. **Create finance processing scripts** following legal script patterns
2. **Add finance data handling** for various document formats
3. **Implement batch processing** for large document sets

### Phase 4: Validation & Testing
1. **Test with sample documents** (10-K, 10-Q, earnings calls)
2. **Validate extraction accuracy** against known ground truth
3. **Performance testing** with large document collections

## Benefits of This Structure

1. **Code Reuse**: Core pipeline components shared between domains
2. **Domain Isolation**: Legal and finance logic separated but consistent
3. **Scalability**: Easy to add new document types (research, compliance)
4. **Maintainability**: Clear separation of concerns
5. **Flexibility**: Domain-specific customization without breaking core logic

This structure allows you to leverage all the existing workflow benefits while cleanly separating finance-specific logic from the legal implementation.