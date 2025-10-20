#!/usr/bin/env python3
"""
Download LiveBench data from HuggingFace and convert to OpenAI batch format using pandas.
Saves JSONL files to test with AI-Flux.
"""

import json
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# LiveBench categories
CATEGORIES = [
    "coding",
    "data_analysis", 
    "instruction_following",
    "math",
    "reasoning",
    "language",
]

# System prompts by category
SYSTEM_PROMPTS = {
    "coding": "You are an expert programmer. Provide accurate, well-structured code solutions.",
    "math": "You are a mathematics expert. Think step by step and provide precise solutions.",
    "data_analysis": "You are a data analysis expert. Provide accurate and detailed analysis.",
    "reasoning": "You are an expert at logical reasoning. Think through problems systematically.",
    "instruction_following": "You are a helpful assistant that follows instructions precisely.",
    "language": "You are an expert in language and communication.",
}

def convert_to_openai_messages(row, system_prompt):
    """Convert a LiveBench row to OpenAI messages format."""
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add all turns as user messages
    for turn in row['turns']:
        messages.append({
            "role": "user",
            "content": turn
        })
    
    return messages

def create_openai_batch_row(row, system_prompt, model="gpt-4o", temperature=0.1, max_tokens=2000):
    """Create an OpenAI batch API request from a LiveBench row."""
    messages = convert_to_openai_messages(row, system_prompt)
    
    # Create custom_id: category_task_question_id
    custom_id = f"{row['category']}_{row['task']}_{row['question_id'][:16]}"
    
    # Convert timestamps and other non-serializable types to strings
    ground_truth = row['ground_truth']
    if isinstance(ground_truth, (list, dict)):
        ground_truth = json.dumps(ground_truth)
    else:
        ground_truth = str(ground_truth)
    
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        # Store metadata for later scoring
        "metadata": {
            "question_id": str(row['question_id']),
            "category": str(row['category']),
            "task": str(row['task']),
            "ground_truth": ground_truth,
            "livebench_release_date": str(row['livebench_release_date'])
        }
    }

def download_and_convert_category(category_name, model="gpt-4o", temperature=0.1, max_tokens=2000):
    """Download a category and convert to both parquet and OpenAI batch format."""
    print(f"\n{'='*60}")
    print(f"Processing category: {category_name}")
    print(f"{'='*60}")
    
    # Load dataset from HuggingFace
    print(f"Loading {category_name} from HuggingFace...")
    dataset = load_dataset(f"livebench/{category_name}", split="test")
    
    # Convert to pandas DataFrame
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} questions")
    
    # Get system prompt
    system_prompt = SYSTEM_PROMPTS.get(category_name, "You are a helpful assistant.")
    
    # Create output directories
    openai_dir = "test_data/"
    os.makedirs(openai_dir, exist_ok=True)
    
    
    # Convert to OpenAI batch format
    print("Converting to OpenAI  JSONL format...")
    openai_requests = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        openai_request = create_openai_batch_row(
            row, 
            system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        openai_requests.append(openai_request)
    
    # Save OpenAI batch requests as JSONL
    openai_path = os.path.join(openai_dir, f"{category_name}_batch.jsonl")
    with open(openai_path, 'w') as f:
        for request in openai_requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"✓ Saved OpenAI batch: {openai_path}")
    
    # Print task distribution
    task_counts = df['task'].value_counts()
    print(f"\nTask distribution in {category_name}:")
    for task, count in task_counts.items():
        print(f"  - {task}: {count} questions")
    
    return len(df)

def main(model="gpt-4o", temperature=0.1, max_tokens=2000):
    """Download and convert all LiveBench categories."""
    print("="*60)
    print("LiveBench Download & JSONL Converter")
    print("="*60)
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    
    total_questions = 0
    results = {}
    
    for category in CATEGORIES:
        try:
            count = download_and_convert_category(
                category, 
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            total_questions += count
            results[category] = {"status": "success", "count": count}
        except Exception as e:
            print(f"✗ Failed to process {category}: {e}")
            results[category] = {"status": "failed", "error": str(e)}
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total questions processed: {total_questions}")
    print(f"\nResults by category:")
    for category, result in results.items():
        if result["status"] == "success":
            print(f"  ✓ {category}: {result['count']} questions")
        else:
            print(f"  ✗ {category}: FAILED - {result['error']}")
    
    print(f"\nData saved in:")
    print(f"  - JSONL files: test_data/")
    
    print("\n" + "="*60)
    print("You can now use the JSONL files with OpenAI Batch API!")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download LiveBench and convert to OpenAI batch format")
    parser.add_argument("--model", default="gpt-4o", help="Model name for OpenAI API")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Max tokens for response")
    
    args = parser.parse_args()
    
    main(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
