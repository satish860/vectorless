"""
Document chunking module for handling large financial documents.
Splits documents at ~100K tokens aligned to heading boundaries.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    content: str
    start_line: int
    end_line: int
    chunk_number: int
    total_chunks: int
    heading_context: str  # The heading this chunk starts under


class DocumentChunker:
    """Handles intelligent chunking of large documents."""
    
    def __init__(self, max_tokens: int = 100000):
        """
        Initialize chunker with token limit.
        
        Args:
            max_tokens: Approximate maximum tokens per chunk (default: 100K)
        """
        self.max_tokens = max_tokens
        # Rough approximation: 1 token ≈ 4 characters
        self.max_chars = max_tokens * 4
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from character count.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def find_headings(self, content: str) -> List[Dict]:
        """
        Find all markdown headings in the document.
        
        Args:
            content: Document content
            
        Returns:
            List of heading info with line numbers and levels
        """
        lines = content.split('\n')
        headings = []
        
        for i, line in enumerate(lines):
            # Match markdown headings (# ## ### etc.)
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line.strip())
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                headings.append({
                    'line_number': i,
                    'level': level,
                    'title': title,
                    'full_line': line
                })
        
        return headings
    
    def find_best_split_point(self, content: str, target_char: int) -> int:
        """
        Find the best heading-aligned split point near the target character.
        
        Args:
            content: Document content
            target_char: Target character position for split
            
        Returns:
            Line number for optimal split point
        """
        lines = content.split('\n')
        
        # Find target line based on character count
        char_count = 0
        target_line = 0
        
        for i, line in enumerate(lines):
            if char_count >= target_char:
                target_line = i
                break
            char_count += len(line) + 1  # +1 for newline
        
        # Find headings around the target line
        headings = self.find_headings(content)
        
        if not headings:
            # No headings found, split at target line
            return target_line
        
        # Find the best heading to split at
        best_heading = None
        min_distance = float('inf')
        
        for heading in headings:
            distance = abs(heading['line_number'] - target_line)
            
            # Prefer higher-level headings (lower level number = higher level)
            # Weight distance by heading level (prefer level 1,2 over 3,4,5,6)
            weighted_distance = distance * (heading['level'] / 2)
            
            if weighted_distance < min_distance and heading['line_number'] <= target_line + 50:
                min_distance = weighted_distance
                best_heading = heading
        
        return best_heading['line_number'] if best_heading else target_line
    
    def get_heading_context(self, content: str, line_number: int) -> str:
        """
        Get the heading context for a given line number.
        
        Args:
            content: Document content
            line_number: Line number to get context for
            
        Returns:
            Context string describing the heading hierarchy
        """
        headings = self.find_headings(content)
        
        # Find all headings before this line
        relevant_headings = [h for h in headings if h['line_number'] <= line_number]
        
        if not relevant_headings:
            return "Document Start"
        
        # Build context from heading hierarchy
        context_parts = []
        current_levels = {}
        
        for heading in relevant_headings:
            level = heading['level']
            title = heading['title']
            
            # Clear deeper levels when we hit a higher level heading
            current_levels = {l: t for l, t in current_levels.items() if l < level}
            current_levels[level] = title
        
        # Build context string from hierarchy
        for level in sorted(current_levels.keys()):
            context_parts.append(f"{'#' * level} {current_levels[level]}")
        
        return " → ".join(context_parts) if context_parts else "Document Start"
    
    def chunk_document(self, content: str) -> List[DocumentChunk]:
        """
        Split document into chunks at heading boundaries.
        
        Args:
            content: Document content to chunk
            
        Returns:
            List of DocumentChunk objects
        """
        # Check if document needs chunking
        estimated_tokens = self.estimate_tokens(content)
        
        if estimated_tokens <= self.max_tokens:
            # Document is small enough, return as single chunk
            lines = content.split('\n')
            return [DocumentChunk(
                content=content,
                start_line=0,
                end_line=len(lines) - 1,
                chunk_number=1,
                total_chunks=1,
                heading_context=self.get_heading_context(content, 0)
            )]
        
        print(f"Document has ~{estimated_tokens:,} tokens, chunking into ~{self.max_tokens:,} token pieces...")
        
        chunks = []
        lines = content.split('\n')
        total_lines = len(lines)
        current_start = 0
        chunk_number = 1
        
        while current_start < total_lines:
            # Calculate target end based on character limit
            remaining_content = '\n'.join(lines[current_start:])
            
            if self.estimate_tokens(remaining_content) <= self.max_tokens:
                # Remaining content fits in one chunk
                chunk_end = total_lines - 1
            else:
                # Find optimal split point
                target_chars = self.max_chars
                split_line = self.find_best_split_point(remaining_content, target_chars)
                chunk_end = current_start + split_line - 1
                
                # Ensure we make progress (at least 10 lines per chunk)
                if chunk_end <= current_start + 10:
                    chunk_end = current_start + max(10, total_lines // 10)
            
            # Create chunk
            chunk_content = '\n'.join(lines[current_start:chunk_end + 1])
            heading_context = self.get_heading_context('\n'.join(lines), current_start)
            
            chunks.append(DocumentChunk(
                content=chunk_content,
                start_line=current_start,
                end_line=chunk_end,
                chunk_number=chunk_number,
                total_chunks=0,  # Will be set after all chunks are created
                heading_context=heading_context
            ))
            
            print(f"Chunk {chunk_number}: Lines {current_start}-{chunk_end} (~{self.estimate_tokens(chunk_content):,} tokens)")
            print(f"  Context: {heading_context}")
            
            current_start = chunk_end + 1
            chunk_number += 1
        
        # Set total chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
        
        print(f"Document split into {total_chunks} chunks")
        return chunks
    
    def validate_chunks(self, chunks: List[DocumentChunk], original_content: str) -> bool:
        """
        Validate that chunks reconstruct the original document.
        
        Args:
            chunks: List of document chunks
            original_content: Original document content
            
        Returns:
            True if validation passes
        """
        if not chunks:
            return False
        
        # Reconstruct content from chunks
        original_lines = original_content.split('\n')
        reconstructed_lines = []
        
        for chunk in sorted(chunks, key=lambda x: x.start_line):
            chunk_lines = chunk.content.split('\n')
            
            # Validate line ranges
            expected_lines = original_lines[chunk.start_line:chunk.end_line + 1]
            
            if chunk_lines != expected_lines:
                print(f"Validation failed for chunk {chunk.chunk_number}")
                return False
            
            reconstructed_lines.extend(chunk_lines)
        
        # Check if reconstruction matches original
        if reconstructed_lines == original_lines:
            print("✅ Chunk validation passed - perfect reconstruction")
            return True
        else:
            print("❌ Chunk validation failed - reconstruction mismatch")
            return False


def demo_chunking():
    """Demonstrate document chunking functionality."""
    
    # Test with a sample large document
    sample_content = """# Main Document

## Section 1: Introduction
This is a sample document to demonstrate chunking.

### Subsection 1.1
Some content here.

### Subsection 1.2  
More content here.

## Section 2: Analysis
This is another major section.

### Subsection 2.1
Analysis content.

## Section 3: Conclusion
Final section content.
"""
    
    # Repeat content to make it larger for demo
    large_content = sample_content * 100  # Make it large enough to require chunking
    
    chunker = DocumentChunker(max_tokens=500)  # Small limit for demo
    chunks = chunker.chunk_document(large_content)
    
    print(f"\nOriginal document: ~{chunker.estimate_tokens(large_content):,} tokens")
    print(f"Split into {len(chunks)} chunks:")
    
    for chunk in chunks:
        print(f"\nChunk {chunk.chunk_number}/{chunk.total_chunks}:")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"  Tokens: ~{chunker.estimate_tokens(chunk.content):,}")
        print(f"  Context: {chunk.heading_context}")
        print(f"  Preview: {chunk.content[:100]}...")
    
    # Validate chunks
    chunker.validate_chunks(chunks, large_content)


if __name__ == "__main__":
    demo_chunking()