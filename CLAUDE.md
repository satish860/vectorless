# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

"vectorless" is a Python project for building a RAG (Retrieval-Augmented Generation) application that doesn't rely on vector embeddings. The project includes the CUAD (Contract Understanding Atticus Dataset) for legal document analysis and Q&A tasks.

## Architecture

- **main.py**: Entry point with basic setup
- **data/**: Contains JSON datasets including:
  - `CUADv1.json`: Contract Understanding Atticus Dataset with legal contract Q&A pairs
  - `test.json` and `train_separate_questions.json`: Training and test datasets
- **sample_dataset/**: Contains sample data for development and testing:
  - `sample_cuad.json`: First item from CUAD dataset with complete contract and 41 Q&A pairs
- **pyproject.toml**: Project configuration using modern Python packaging standards

## Development Commands

This project uses `uv` for Python package management instead of pip.

### Project Setup
```bash
# Install dependencies
uv sync

# Add new dependencies
uv add <package-name>

# Add development dependencies
uv add --dev <package-name>
```

### Running the Application
```bash
# Run with uv
uv run python main.py

# Or activate the environment and run directly
uv shell
python main.py
```

## Data Structure

The CUAD dataset contains legal contract analysis data structured as:
- Documents with titles referencing contract types and dates
- Question-answer pairs for contract review targeting specific legal clauses
- Answers include text snippets and their positions in source documents
- Designed for training models to identify contract elements requiring legal review

## RAG Implementation Goals

This project aims to implement retrieval-augmented generation without traditional vector embeddings, exploring alternative approaches to document retrieval and question answering for legal contract analysis.

## Current Context

Current year: September 2025