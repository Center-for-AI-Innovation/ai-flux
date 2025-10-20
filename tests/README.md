# Test Inference Script

Submit SLURM jobs to run model inference on test data. Each test file gets a separate SLURM job for parallel processing.

## Quick Start

```bash
# Run all tests with default model (llama3.2:3b)
python tests/test_inference.py

# Test one category
python tests/test_inference.py --category language

# Different model
python tests/test_inference.py llama3.3:70b

# Custom SLURM settings
python tests/test_inference.py --mem 32G --batch-size 8 --temperature 0.7
```

## Test Categories

| Category | Items | Output | Command |
|----------|-------|--------|---------|
| `language` | 190 | `results/language_results.json` | `python tests/test_inference.py --category language` |
| `reasoning` | 200 | `results/reasoning_results.json` | `python tests/test_inference.py --category reasoning` |
| `math` | 368 | `results/math_results.json` | `python tests/test_inference.py --category math` |
| `data_analysis` | 150 | `results/data_analysis_results.json` | `python tests/test_inference.py --category data_analysis` |

**Total: 908 test items** • Omit `--category` to run all

## Usage

```bash
python tests/test_inference.py [MODEL] [OPTIONS]
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `MODEL` | `llama3.2:3b` | Model name (positional arg) |
| `--category` | all | Filter by category: `language`, `math`, `reasoning`, `data_analysis` |
| `--account` | `rohan13-ic` | SLURM account |
| `--partition` | `IllinoisComputes-GPU` | SLURM partition |
| `--mem` | `16G` | Memory allocation |
| `--nodes` | `1` | Number of nodes |
| `--engine` | `ollama` | Inference engine: `ollama` or `vllm` |
| `--batch-size` | `4` | Items per batch |
| `--save-frequency` | `50` | Save every N items |
| `--temperature` | - | Sampling temperature (0.0-1.0) |
| `--max-tokens` | - | Maximum tokens to generate |
| `--top-p` | - | Top-p sampling |
| `--top-k` | - | Top-k sampling |

<details>
<summary><b>All Options</b></summary>

**SLURM:**
- `--account`, `--partition`, `--nodes`, `--mem`, `--gpus-per-node`, `--time`, `--cpus-per-task`, `--engine`

**Data:**
- `--category`, `--test-data-dir`, `--output-dir`

**Processing:**
- `--batch-size`, `--save-frequency`, `--max-retries`, `--retry-delay`

**Model:**
- `--temperature`, `--max-tokens`, `--top-p`, `--top-k`

</details>

## Examples

```bash
# Test single category with custom model
python tests/test_inference.py qwen2.5:7b --category math

# More memory and larger batches
python tests/test_inference.py --mem 32G --batch-size 8

# Custom model parameters
python tests/test_inference.py --temperature 0.1 --max-tokens 2000

# Different SLURM account/partition
python tests/test_inference.py --account myaccount --partition mypartition

# Use vLLM engine
python tests/test_inference.py --engine vllm --mem 32G

# Show all options
python tests/test_inference.py --help
```

## Workflow

**Quick test → Verify → Full run:**

```bash
# 1. Test smallest category first (150 items)
python tests/test_inference.py --category data_analysis

# 2. Check results
cat results/data_analysis_results.json | head -50

# 3. Run all categories
python tests/test_inference.py
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# View logs
ls .aiflux/logs/
tail -f .aiflux/logs/<job_id>.out

# Cancel jobs
scancel <job_id>              # Cancel specific job
scancel -u $USER              # Cancel all your jobs
```

## Output Format

Results saved as JSON arrays in `results/`:

```json
[
  {
    "input": {
      "custom_id": "...",
      "body": {
        "model": "gpt-4o",
        "messages": [...],
        "temperature": 0.1
      },
      "metadata": {...}
    },
    "output": {
      "id": "...",
      "model": "llama3.2:3b",
      "choices": [{
        "message": {
          "role": "assistant",
          "content": "Model response..."
        }
      }]
    },
    "metadata": {
      "model": "llama3.2:3b",
      "timestamp": "2025-10-20T13:00:00",
      ...
    }
  }
]
```

## Comparison with aiflux CLI

This script automates multiple `aiflux run` commands:

```bash
# Instead of running this for each file:
aiflux run --model llama3.2:3b --input data/prompts.jsonl --output results/output.json \
  --account rohan13-ic --partition IllinoisComputes-GPU --nodes 1 --mem 16G

# Just run this once:
python tests/test_inference.py
```
