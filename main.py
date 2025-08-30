"""
Vectorless RAG Demo - Document Segmentation with GPT-OSS-120B
"""

from src.legal_segmenter import demo_segmentation


def main():
    print("Vectorless RAG - Document Segmentation Demo")
    print("Using OpenRouter GPT-OSS-120B for intelligent legal contract segmentation")
    print("=" * 80)
    
    # Run the document segmentation demo
    demo_segmentation()
    
    print("\n" + "=" * 80)
    print("Demo completed! Check the sections above to see how the legal contract")
    print("was intelligently segmented into meaningful clauses by GPT-OSS-120B.")


if __name__ == "__main__":
    main()
