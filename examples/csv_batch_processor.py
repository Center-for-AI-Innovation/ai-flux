#!/usr/bin/env python3
"""Example script for CSV batch processing with AI-Flux."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.aiflux import BatchProcessor, SlurmRunner
from src.aiflux.core.config import Config
from src.aiflux.io import CSVSinglePromptHandler, JSONOutputHandler

def process_csv_data():
    """Example of processing a CSV file."""
    # Load model configuration
    config = Config()
    model_config = config.get_model_config("llama3")
    
    # Define a system prompt for context
    system_prompt = "You are a research assistant specializing in summarizing scientific papers."
    
    # Initialize processor with CSV handler
    processor = BatchProcessor(
        model_config=model_config,
        input_handler=CSVSinglePromptHandler(),
        output_handler=JSONOutputHandler(),
        batch_size=4
    )
    
    # Setup SLURM configuration
    slurm_config = config.get_slurm_config()
    slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
    slurm_config.time = "02:00:00"
    
    # Process inputs with template and system prompt
    runner = SlurmRunner(processor, slurm_config)
    runner.run(
        input_path='data/papers.csv',
        output_path='results/paper_summaries.json',
        prompt_template=(
            "Please summarize the following research paper:\n\n"
            "Title: {title}\n"
            "Abstract: {abstract}\n\n"
            "Provide a concise summary focusing on:\n"
            "1. Main research question\n"
            "2. Key methodology\n"
            "3. Main findings\n"
            "4. Significance of results"
        ),
        system_prompt=system_prompt
    )

def create_example_csv_data():
    """Create example CSV data for batch processing."""
    # Create example data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create example CSV input
    csv_input = """title,abstract
Quantum Supremacy Using a Programmable Superconducting Processor,"The promise of quantum computers is that certain computational tasks might be executed exponentially faster on a quantum processor than on a classical processor. A fundamental challenge is to build a high-fidelity processor capable of running quantum algorithms in an exponentially large computational space."
Large Language Models in Machine Learning,"Recent advances in transformer architectures and pre-training techniques have led to significant improvements in natural language processing tasks. This paper surveys the current state of large language models and their applications."
"""
    
    with open(data_dir / 'papers.csv', 'w') as f:
        f.write(csv_input)

if __name__ == '__main__':
    create_example_csv_data()
    print("Processing CSV data...")
    process_csv_data() 