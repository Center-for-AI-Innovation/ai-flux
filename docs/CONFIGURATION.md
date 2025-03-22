# AI-Flux Configuration Guide

AI-Flux uses a flexible configuration system with clear parameter priority to ensure a smooth user experience whether using code, environment variables, or defaults.

## Configuration Priority System

When determining which configuration value to use, AI-Flux follows this priority order:

1. **Direct code parameters** (highest priority)
2. **Environment variables** (.env file)
3. **Default values** (lowest priority)

This means you can override any setting by explicitly specifying it in your code, and the system will fall back to environment variables and defaults as needed.

## Core Configuration Parameters

The tables below show each parameter with its environment variable name, code setting, and default value.

### SLURM Configuration

| Parameter | Environment Variable | Code Setting | Default | Description |
|-----------|----------------------|-------------|---------|-------------|
| Account | `SLURM_ACCOUNT` | `slurm_config.account = "myaccount"` | (required) | SLURM account name with GPU access |
| Partition | `SLURM_PARTITION` | `slurm_config.partition = "a100"` | `gpuA100x4` | GPU partition to use |
| Time | `SLURM_TIME` | `slurm_config.time = "01:00:00"` | `00:30:00` | Job time limit (HH:MM:SS) |
| Memory | `SLURM_MEM` | `slurm_config.mem = "32G"` | `32G` | Memory allocation per node |
| GPUs per node | `SLURM_GPUS_PER_NODE` | `slurm_config.gpus_per_node = 2` | `1` | Number of GPUs to allocate |
| Nodes | `SLURM_NODES` | `slurm_config.nodes = 1` | `1` | Number of nodes to request |
| CPUs per task | `SLURM_CPUS_PER_TASK` | `slurm_config.cpus_per_task = 8` | `4` | CPUs per task |

### Processing Configuration

| Parameter | Environment Variable | Code Setting | Default | Description |
|-----------|----------------------|-------------|---------|-------------|
| Model name | `MODEL_NAME` | `model="llama3.2:3b"` (in `run()`) | `llama3.2:3b` | Model to use (format: `model:size`) |
| Batch size | `BATCH_SIZE` | `batch_size=8` (in `run()`) | `4` | Number of prompts to process in parallel |
| Save frequency | `SAVE_FREQUENCY` | `save_frequency=100` (in `run()`) | `50` | How often to save intermediate results |
| Temperature | `TEMPERATURE` | `temperature=0.8` (in `run()`) | `0.7` | Sampling temperature |
| Max tokens | `MAX_TOKENS` | `max_tokens=4096` (in `run()`) | `2048` | Maximum tokens to generate |
| Top P | `TOP_P` | `top_p=0.95` (in `run()`) | `0.9` | Top-p sampling parameter |
| Top K | `TOP_K` | `top_k=50` (in `run()`) | `40` | Top-k sampling parameter |

### Directory Configuration

| Parameter | Environment Variable | Code Setting | Default | Description |
|-----------|----------------------|-------------|---------|-------------|
| Workspace | `WORKSPACE` | `workspace="/path/to/workspace"` in SlurmRunner | `./` | Project root directory |
| Input directory | `DATA_INPUT_DIR` | Set via config | `data/input` | Directory for input files |
| Output directory | `DATA_OUTPUT_DIR` | Set via config | `data/output` | Directory for output files |

## Configuration Methods

You can configure AI-Flux in multiple ways, depending on your preference and needs.

### Method 1: Environment Variables (.env file)

The simplest way to configure AI-Flux is by creating a `.env` file in your project root:

```bash
# SLURM Settings
SLURM_ACCOUNT=my-account
SLURM_PARTITION=gpuA100x4
SLURM_TIME=01:00:00

# Model Settings
MODEL_NAME=gemma3:27b
BATCH_SIZE=8
MAX_TOKENS=4096
```

### Method 2: Direct Code Parameters (Recommended)

For more control, you can set configuration parameters directly in your code:

```python
from aiflux.slurm import SlurmRunner
from aiflux.core.config import Config

# Setup SLURM configuration
config = Config()
slurm_config = config.get_slurm_config()
slurm_config.account = "my-account"
slurm_config.partition = "a100"
slurm_config.time = "02:00:00"
slurm_config.mem = "64G"

# Submit job with additional processing options
runner = SlurmRunner(config=slurm_config)
job_id = runner.run(
    input_path="large_dataset.jsonl",
    output_path="results.json",
    model="llama3.2:70b",
    batch_size=2,         # Process 2 items at a time
    save_frequency=100,   # Save intermediate results every 100 items
    temperature=0.8,      # Model temperature
    max_tokens=4096       # Maximum tokens to generate
)
```

For details on model-specific requirements and recommendations, see the [Models Guide](MODELS.md). 