#!/usr/bin/env python3
"""Example script for batch processing with AI-Flux."""

import os
import sys
import json
import datetime
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.aiflux import BatchProcessor, SlurmRunner
from src.aiflux.core.config import Config
from src.aiflux.io import JSONBatchHandler, CSVSinglePromptHandler, JSONOutputHandler
from examples.utils import get_timestamped_filename, ensure_results_dir

def process_json_batch():
    """Example of processing a JSON batch file."""
    # Load model configuration
    config = Config()
    model_config = config.load_model_config("mistral", "8x7b")
    
    # Initialize processor with JSON handler
    processor = BatchProcessor(
        model_config=model_config,
        input_handler=JSONBatchHandler(),
        output_handler=JSONOutputHandler(),
        batch_size=8
    )
    
    # Setup SLURM configuration
    slurm_config = config.get_slurm_config()
    slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
    slurm_config.time = "01:00:00"
    
    # Create timestamped output path
    output_path = get_timestamped_filename('results/batch_results.json')
    
    # Run on SLURM
    runner = SlurmRunner(config=slurm_config)
    runner.run(
        processor=processor,
        input_source='data/prompts.json',
        output_path=output_path
    )
    
    print(f"Results saved to: {output_path}")

def process_csv_data():
    """Example of processing a CSV file."""
    # Load model configuration
    config = Config()
    model_config = config.load_model_config("llama3.2", "11b")
    
    # Define prompt template
    prompt_template = (
        "Please summarize the following research paper:\n\n"
        "Title: {title}\n"
        "Abstract: {abstract}\n\n"
        "Provide a concise summary focusing on:\n"
        "1. Main research question\n"
        "2. Key methodology\n"
        "3. Main findings\n"
        "4. Significance of results"
    )
    
    # Define system prompt
    system_prompt = "You are a scientific research assistant with expertise in summarizing academic papers."
    
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
    
    # Create timestamped output path
    output_path = get_timestamped_filename('results/paper_summaries.json')
    
    # Run on SLURM with both template and system prompt
    runner = SlurmRunner(config=slurm_config)
    runner.run(
        processor=processor,
        input_source='data/papers.csv',
        output_path=output_path,
        prompt_template=prompt_template,
        system_prompt=system_prompt
    )
    
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    # Create example data directory and results directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Ensure results directory exists
    ensure_results_dir()
    
    # Create example JSON input with OpenAI-compatible format
    json_input = """[
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain quantum computing in simple terms."
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in healthcare technologies."
                },
                {
                    "role": "user",
                    "content": "What are the main applications of machine learning in healthcare?"
                }
            ],
            "temperature": 0.8,
            "max_tokens": 2048
        }
    ]"""
    
    with open(data_dir / 'prompts.json', 'w') as f:
        f.write(json_input)
    
    # Create example CSV input
    csv_input = """title,abstract
Quantum Supremacy Using a Programmable Superconducting Processor,"The promise of quantum computers is that certain computational tasks might be executed exponentially faster on a quantum processor than on a classical processor. A fundamental challenge is to build a high-fidelity processor capable of running quantum algorithms in an exponentially large computational space."
Large Language Models in Machine Learning,"Recent advances in transformer architectures and pre-training techniques have led to significant improvements in natural language processing tasks. This paper surveys the current state of large language models and their applications."
"""
    
    with open(data_dir / 'papers.csv', 'w') as f:
        f.write(csv_input)
    
    # Run examples
    print("Processing JSON batch...")
    process_json_batch()
    
    print("\nProcessing CSV data...")
    process_csv_data() 