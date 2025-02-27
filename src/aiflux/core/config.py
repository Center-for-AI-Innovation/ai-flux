#!/usr/bin/env python3
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator

class ResourceConfig(BaseModel):
    """Model resource configuration."""
    gpu_layers: int = Field(..., ge=1)
    gpu_memory: str = Field(..., pattern=r"^\d+GB$")
    batch_size: int = Field(..., ge=1)
    max_concurrent: int = Field(..., ge=1)

class ParameterConfig(BaseModel):
    """Model parameter configuration."""
    temperature: float = Field(..., ge=0.0, le=1.0)
    top_p: float = Field(..., ge=0.0, le=1.0)
    max_tokens: int = Field(..., ge=1)
    stop_sequences: List[str]

class SystemConfig(BaseModel):
    """Model system configuration."""
    prompt: str

class ValidationConfig(BaseModel):
    """Model validation configuration."""
    temperature_range: List[float] = Field(..., min_items=2, max_items=2)
    max_tokens_limit: int = Field(..., ge=1)
    batch_size_range: List[int] = Field(..., min_items=2, max_items=2)
    concurrent_range: List[int] = Field(..., min_items=2, max_items=2)

    @field_validator("temperature_range")
    @classmethod
    def validate_temperature_range(cls, v):
        if not (0.0 <= v[0] <= v[1] <= 1.0):
            raise ValueError("Temperature range must be between 0.0 and 1.0")
        return v

    @field_validator("batch_size_range", "concurrent_range")
    @classmethod
    def validate_range(cls, v):
        if not (v[0] <= v[1]):
            raise ValueError("Range start must be less than or equal to end")
        return v

class RequirementsConfig(BaseModel):
    """Model requirements configuration."""
    min_gpu_memory: str = Field(..., pattern=r"^\d+GB$")
    recommended_gpu: str
    cuda_version: str = Field(..., pattern=r"^>=\d+\.\d+$")
    cpu_threads: int = Field(..., ge=1)
    gpu_memory_utilization: float = Field(..., ge=0.0, le=1.0)

class ModelConfig(BaseModel):
    """Complete model configuration."""
    name: str = Field(..., pattern=r"^[a-zA-Z0-9.-]+:\d+b$")
    resources: ResourceConfig
    parameters: ParameterConfig
    system: SystemConfig
    validation: ValidationConfig
    requirements: RequirementsConfig

class SlurmConfig(BaseModel):
    """SLURM configuration with env var support."""
    account: str = Field(default_factory=lambda: os.getenv('SLURM_ACCOUNT'))
    partition: str = Field(
        default_factory=lambda: os.getenv('SLURM_PARTITION', 'a100')
    )
    nodes: int = Field(
        default_factory=lambda: int(os.getenv('SLURM_NODES', '1'))
    )
    gpus_per_node: int = Field(
        default_factory=lambda: int(os.getenv('SLURM_GPUS_PER_NODE', '1'))
    )
    time: str = Field(
        default_factory=lambda: os.getenv('SLURM_TIME', '00:30:00')
    )
    memory: str = Field(
        default_factory=lambda: os.getenv('SLURM_MEM', '32G')
    )
    cpus_per_task: int = Field(
        default_factory=lambda: int(os.getenv('SLURM_CPUS_PER_TASK', '4'))
    )

def parse_gpu_memory(memory_str: str) -> int:
    """Convert GPU memory string to GB value."""
    match = re.match(r"^(\d+)GB$", memory_str)
    if not match:
        raise ValueError(f"Invalid GPU memory format: {memory_str}")
    return int(match.group(1))

class Config:
    """Central configuration management."""
    
    def __init__(self):
        self.package_dir = Path(__file__).parent.parent
        self.templates_dir = self.package_dir / 'templates'
        
        # Load environment variables from .env file if it exists
        self._load_env_file()
        
        # Initialize workspace paths
        self.workspace = Path.cwd()
        
        # Define default paths
        self.default_paths = {
            'DATA_INPUT_DIR': self.workspace / "data" / "input",
            'DATA_OUTPUT_DIR': self.workspace / "data" / "output",
            'MODELS_DIR': self.workspace / "models",
            'LOGS_DIR': self.workspace / "logs",
            'CONTAINERS_DIR': self.workspace / "containers",
            'APPTAINER_TMPDIR': self.workspace / "tmp",
            'APPTAINER_CACHEDIR': self.workspace / "tmp" / "cache",
            'OLLAMA_HOME': self.workspace / ".ollama",
        }
        
        # Define default settings
        self.default_settings = {
            'SLURM_PARTITION': 'a100',
            'SLURM_NODES': '1',
            'SLURM_GPUS_PER_NODE': '1',
            'SLURM_TIME': '00:30:00',
            'SLURM_MEM': '32G',
            'SLURM_CPUS_PER_TASK': '4',
            'OLLAMA_ORIGINS': '*',
            'OLLAMA_INSECURE': 'true',
            'CURL_CA_BUNDLE': '',
            'SSL_CERT_FILE': '',
        }
    
    def _load_env_file(self):
        """Load environment variables from .env file in project root."""
        # Try to find .env file in current directory or parent directories
        current_dir = Path.cwd()
        env_file = current_dir / '.env'
        
        # Check up to 3 parent directories
        for _ in range(3):
            if env_file.exists():
                self._load_env_from_file(env_file)
                return
            
            # Move up one directory
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root directory
                break
            current_dir = parent_dir
            env_file = current_dir / '.env'
    
    def _load_env_from_file(self, env_file: Path):
        """Load environment variables from the specified .env file.
        
        Environment variables are loaded with the following precedence:
        1. Existing environment variables (highest priority)
        2. Variables from .env file (middle priority)
        3. Default values (lowest priority)
        """
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key-value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove inline comments if present
                        if '#' in value:
                            value = value.split('#', 1)[0].strip()
                        
                        # Only set if not already in environment and value is not empty
                        if key and value and key not in os.environ:
                            os.environ[key] = value
        except Exception as e:
            # Log the error but continue execution
            import logging
            logging.warning(f"Error loading .env file: {str(e)}")
            pass
    
    def get_path(self, path_name: str, code_path: Optional[Union[str, Path]] = None) -> Path:
        """Get a resolved path following precedence: code path > env var > default.
        
        Args:
            path_name: Name of the path (e.g., 'DATA_INPUT_DIR')
            code_path: Optional explicit path from code
            
        Returns:
            Resolved Path object
        """
        # 1. Code path (highest priority)
        if code_path is not None:
            return Path(code_path)
        
        # 2. Environment variable (middle priority)
        if path_name in os.environ and os.environ[path_name]:
            return Path(os.environ[path_name])
        
        # 3. Default value (lowest priority)
        if path_name in self.default_paths:
            return self.default_paths[path_name]
        
        # Fallback to workspace if no match
        return self.workspace / path_name.lower()
    
    def get_setting(self, setting_name: str, code_value: Optional[Any] = None) -> Any:
        """Get a resolved setting following precedence: code value > env var > default.
        
        Args:
            setting_name: Name of the setting (e.g., 'SLURM_PARTITION')
            code_value: Optional explicit value from code
            
        Returns:
            Resolved setting value
        """
        # 1. Code value (highest priority)
        if code_value is not None:
            return code_value
        
        # 2. Environment variable (middle priority)
        if setting_name in os.environ and os.environ[setting_name]:
            return os.environ[setting_name]
        
        # 3. Default value (lowest priority)
        if setting_name in self.default_settings:
            return self.default_settings[setting_name]
        
        # Return None if no match
        return None
    
    def ensure_directory(self, path: Path) -> Path:
        """Ensure a directory exists.
        
        Args:
            path: Path to ensure
            
        Returns:
            The same path
        """
        if path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_environment(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Get a complete environment dictionary with all settings.
        
        Args:
            overrides: Optional dictionary of override values
            
        Returns:
            Dictionary of environment variables
        """
        # Start with current environment
        env = os.environ.copy()
        
        # Add all paths
        for path_name in self.default_paths:
            code_path = overrides.get(path_name) if overrides else None
            env[path_name] = str(self.get_path(path_name, code_path))
        
        # Add all settings
        for setting_name in self.default_settings:
            code_value = overrides.get(setting_name) if overrides else None
            env[setting_name] = str(self.get_setting(setting_name, code_value))
        
        # Add SLURM config
        # Create a filtered dictionary with SLURM-specific overrides
        slurm_overrides = {}
        if overrides:
            # Map SLURM_* keys to their corresponding field names in SlurmConfig
            slurm_field_mapping = {
                'SLURM_ACCOUNT': 'account',
                'SLURM_PARTITION': 'partition',
                'SLURM_NODES': 'nodes',
                'SLURM_GPUS_PER_NODE': 'gpus_per_node',
                'SLURM_TIME': 'time',
                'SLURM_MEM': 'memory',
                'SLURM_CPUS_PER_TASK': 'cpus_per_task'
            }
            
            for env_key, field_name in slurm_field_mapping.items():
                if env_key in overrides:
                    slurm_overrides[field_name] = overrides[env_key]
        
        slurm_config = self.get_slurm_config(slurm_overrides)
        env.update({
            'SLURM_ACCOUNT': slurm_config.account,
            'SLURM_PARTITION': slurm_config.partition,
            'SLURM_NODES': str(slurm_config.nodes),
            'SLURM_GPUS_PER_NODE': str(slurm_config.gpus_per_node),
            'SLURM_TIME': slurm_config.time,
            'SLURM_MEM': slurm_config.memory,
            'SLURM_CPUS_PER_TASK': str(slurm_config.cpus_per_task),
        })
        
        # Filter out None values
        return {k: v for k, v in env.items() if v is not None}
    
    def load_model_config(
        self,
        model_type: str,
        model_size: str,
        custom_config_path: Optional[str] = None
    ) -> ModelConfig:
        """Load and validate model configuration.
        
        Args:
            model_type: Type of model (e.g., 'qwen', 'llama')
            model_size: Size of model (e.g., '7b', '70b')
            custom_config_path: Optional path to custom config
            
        Returns:
            Validated ModelConfig
            
        Raises:
            ValueError: If configuration is invalid
        """
        if custom_config_path:
            config_path = Path(custom_config_path)
        else:
            config_path = self.templates_dir / model_type / f"{model_size}.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Validate against schema
            model_config = ModelConfig(**config_data)
            
            # Additional validation
            gpu_mem = parse_gpu_memory(model_config.resources.gpu_memory)
            min_gpu_mem = parse_gpu_memory(model_config.requirements.min_gpu_memory)
            
            if gpu_mem < min_gpu_mem:
                raise ValueError(
                    f"GPU memory ({gpu_mem}GB) less than minimum required "
                    f"({min_gpu_mem}GB)"
                )
            
            if (model_config.resources.batch_size < 
                model_config.validation.batch_size_range[0] or
                model_config.resources.batch_size > 
                model_config.validation.batch_size_range[1]):
                raise ValueError("Batch size outside valid range")
            
            if (model_config.resources.max_concurrent < 
                model_config.validation.concurrent_range[0] or
                model_config.resources.max_concurrent > 
                model_config.validation.concurrent_range[1]):
                raise ValueError("Max concurrent requests outside valid range")
            
            return model_config
            
        except Exception as e:
            raise ValueError(f"Error loading config from {config_path}: {str(e)}")
    
    def get_slurm_config(
        self,
        overrides: Optional[Dict[str, Any]] = None
    ) -> SlurmConfig:
        """Get SLURM configuration with overrides.
        
        Args:
            overrides: Optional configuration overrides
            
        Returns:
            SlurmConfig instance
        """
        config = SlurmConfig()
        if overrides:
            for key, value in overrides.items():
                setattr(config, key, value)
        return config
    
    def to_env_dict(self) -> Dict[str, str]:
        """Convert configurations to environment variables.
        
        Returns:
            Dictionary of environment variables
        """
        slurm_config = self.get_slurm_config()
        
        env_dict = {
            'SLURM_ACCOUNT': slurm_config.account,
            'SLURM_PARTITION': slurm_config.partition,
            'SLURM_NODES': str(slurm_config.nodes),
            'SLURM_GPUS_PER_NODE': str(slurm_config.gpus_per_node),
            'SLURM_TIME': slurm_config.time,
            'SLURM_MEM': slurm_config.memory,
            'SLURM_CPUS_PER_TASK': str(slurm_config.cpus_per_task)
        }
        
        return {k: v for k, v in env_dict.items() if v is not None} 