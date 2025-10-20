#!/usr/bin/env python3
"""
Test inference script for running AI-Flux on test data.

This script processes multiple JSONL test files and generates separate output files
for each input using the AI-Flux library.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
import argparse

from aiflux.processors import BatchProcessor
from aiflux.core.config import Config, ModelConfig
from aiflux.slurm.runner import SlurmRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestInferenceRunner:
    """Runner for batch inference on test data files using SLURM."""
    
    def __init__(
        self,
        model: str = "llama3.2:3b",
        test_data_dir: str = "tests/data/test_data",
        output_dir: str = "results",
        batch_size: int = 4,
        save_frequency: int = 50,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize test inference runner.
        
        Args:
            model: Model name (e.g., "llama3.2:3b")
            test_data_dir: Directory containing test JSONL files
            output_dir: Directory to save output files
            batch_size: Number of items to process in a batch
            save_frequency: How often to save intermediate results
            max_retries: Maximum retry attempts for failed items
            retry_delay: Delay between retries in seconds
        """
        self.model = model
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize config
        self.config = Config()
        
    def get_test_files(self, category: Optional[str] = None) -> List[Path]:
        """
        Get all JSONL test files from the test data directory.
        
        Args:
            category: Optional category filter (e.g., 'language', 'math', 'reasoning', 'data_analysis')
        
        Returns:
            List of paths to test JSONL files
        """
        if not self.test_data_dir.exists():
            raise FileNotFoundError(f"Test data directory not found: {self.test_data_dir}")
        
        all_files = sorted(self.test_data_dir.glob("*.jsonl"))
        
        if not all_files:
            raise FileNotFoundError(f"No JSONL files found in {self.test_data_dir}")
        
        # Filter by category if specified
        if category:
            jsonl_files = [f for f in all_files if category.lower() in f.stem.lower()]
            if not jsonl_files:
                available = [f.stem.replace('_batch', '') for f in all_files]
                raise ValueError(
                    f"No files found for category '{category}'. "
                    f"Available categories: {', '.join(available)}"
                )
        else:
            jsonl_files = all_files
        
        logger.info(f"Found {len(jsonl_files)} test file(s):")
        for f in jsonl_files:
            logger.info(f"  - {f.name}")
        
        return jsonl_files
    
    def get_output_path(self, input_file: Path) -> Path:
        """
        Generate output file path from input file name.
        
        Args:
            input_file: Path to input JSONL file
            
        Returns:
            Path to output JSON file
        """
        # Convert "language_batch.jsonl" to "language_results.json"
        base_name = input_file.stem.replace("_batch", "_results")
        return self.output_dir / f"{base_name}.json"
    
    def run(
        self,
        account: str,
        partition: str,
        category: Optional[str] = None,
        nodes: int = 1,
        gpus_per_node: Optional[int] = None,
        time: Optional[str] = None,
        mem: str = "16G",
        cpus_per_task: Optional[int] = None,
        engine: str = "ollama",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Submit SLURM jobs for inference on all test files.
        Each test file will be submitted as a separate SLURM job.
        
        Args:
            account: SLURM account
            partition: SLURM partition
            category: Optional category filter (e.g., 'language', 'math')
            nodes: Number of nodes
            gpus_per_node: GPUs per node
            time: Time limit
            mem: Memory allocation
            cpus_per_task: CPUs per task
            engine: Engine to use (ollama or vllm)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Dictionary mapping input file names to job IDs
        """
        test_files = self.get_test_files(category=category)
        job_ids = {}
        
        # Setup SLURM config
        from aiflux.core.config import SlurmConfig, EngineConfig
        
        slurm_config_dict = {
            "account": account,
            "partition": partition,
            "nodes": nodes,
            "mem": mem,
        }
        
        if gpus_per_node is not None:
            slurm_config_dict["gpus_per_node"] = gpus_per_node
        if time is not None:
            slurm_config_dict["time"] = time
        if cpus_per_task is not None:
            slurm_config_dict["cpus_per_task"] = cpus_per_task
        
        slurm_config = self.config.get_slurm_config(slurm_config_dict)
        
        # Setup engine config
        if engine == "vllm":
            engine_config = EngineConfig(
                engine="vllm",
                home=str(self.config.workspace / ".vllm")
            )
        else:
            engine_config = EngineConfig(
                engine="ollama",
                home=str(self.config.workspace / ".ollama")
            )
        
        logger.info(f"Submitting SLURM jobs with model: {self.model}")
        logger.info(f"Account: {account}, Partition: {partition}")
        logger.info(f"Processing {len(test_files)} files")
        
        for i, input_file in enumerate(test_files, 1):
            output_file = self.get_output_path(input_file)
            
            logger.info(f"\nSubmitting job {i}/{len(test_files)}: {input_file.name}")
            
            # Initialize runner for this file
            runner = SlurmRunner(config=slurm_config, engine_config=engine_config)
            
            # Build kwargs
            kwargs = {
                "model": self.model,
                "batch_size": self.batch_size,
                "save_frequency": self.save_frequency,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
            }
            
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
            if top_k is not None:
                kwargs["top_k"] = top_k
            
            try:
                # Submit job
                job_id = runner.run(
                    input_path=str(input_file),
                    output_path=str(output_file),
                    **kwargs
                )
                job_ids[input_file.name] = job_id
                logger.info(f"✓ Job submitted: {job_id}")
                logger.info(f"  Output will be saved to: {output_file}")
                
            except Exception as e:
                logger.error(f"✗ Error submitting job for {input_file.name}: {e}")
                job_ids[input_file.name] = None
        
        return job_ids


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs to run inference on test data files. Each test file will be submitted as a separate job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data options
    parser.add_argument(
        "model",
        nargs="?",
        default="llama3.2:3b",
        help="Model name"
    )
    parser.add_argument(
        "--category",
        help="Test specific category only (e.g., 'language', 'math', 'reasoning', 'data_analysis')"
    )
    parser.add_argument(
        "--test-data-dir",
        default="tests/data/test_data",
        help="Directory containing test JSONL files"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save output files"
    )
    
    # SLURM options (required)
    parser.add_argument(
        "--account",
        default="rohan13-ic",
        help="SLURM account"
    )
    parser.add_argument(
        "--partition",
        default="IllinoisComputes-GPU",
        help="SLURM partition"
    )
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--mem", default="16G", help="Memory allocation")
    parser.add_argument("--gpus-per-node", type=int, help="GPUs per node")
    parser.add_argument("--time", help="Time limit (e.g., 01:00:00)")
    parser.add_argument("--cpus-per-task", type=int, help="CPUs per task")
    parser.add_argument(
        "--engine",
        choices=["ollama", "vllm"],
        default="ollama",
        help="Inference engine"
    )
    
    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=50,
        help="Save intermediate results every N items"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts"
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay between retries (seconds)"
    )
    
    # Model parameters
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, help="Top-k sampling parameter")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = TestInferenceRunner(
        model=args.model,
        test_data_dir=args.test_data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        save_frequency=args.save_frequency,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    
    # Submit SLURM jobs
    logger.info("Submitting SLURM jobs for test inference")
    logger.info(f"Model: {args.model}")
    logger.info(f"Account: {args.account}")
    logger.info(f"Partition: {args.partition}")
    if args.category:
        logger.info(f"Category filter: {args.category}")
    
    job_ids = runner.run(
        account=args.account,
        partition=args.partition,
        category=args.category,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        time=args.time,
        mem=args.mem,
        cpus_per_task=args.cpus_per_task,
        engine=args.engine,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    
    logger.info("\n" + "="*60)
    logger.info("JOB SUBMISSION COMPLETE")
    logger.info("="*60)
    for input_name, job_id in job_ids.items():
        if job_id:
            logger.info(f"✓ {input_name} → Job ID: {job_id}")
        else:
            logger.info(f"✗ {input_name} → FAILED")
    
    logger.info("\nMonitor jobs with: squeue -u $USER")
    logger.info("Check logs in: .aiflux/logs/")
    logger.info(f"Results will be saved in: {args.output_dir}/")


if __name__ == "__main__":
    main()

