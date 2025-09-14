import json
import os
import uuid
import copy
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from .base_gnn_model import ModelConfig
from .transductive_model import TransductiveModelConfig
from .inductive_model import InductiveModelConfig


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ConfigTemplate:
    """Template for creating configurations"""
    template_name: str
    base_config: ModelConfig
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    required_parameters: List[str] = field(default_factory=list)
    optional_parameters: List[str] = field(default_factory=list)
    description: str = ""

    def generate_config(self, **overrides) -> ModelConfig:
        """Generate configuration from template with overrides"""
        # Create a copy of base config
        config_dict = asdict(self.base_config)

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(self.base_config, key):
                config_dict[key] = value

        # Recreate config object
        config_class = type(self.base_config)
        return config_class(**config_dict)

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """Validate parameters against template constraints"""
        errors = []

        for param_name, value in parameters.items():
            if param_name in self.parameter_ranges:
                min_val, max_val = self.parameter_ranges[param_name]
                if not (min_val <= value <= max_val):
                    errors.append(f"{param_name} = {value} is outside range [{min_val}, {max_val}]")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)


@dataclass
class ConfigPreset:
    """Predefined configuration preset"""
    preset_name: str
    config: ModelConfig
    use_case: str = ""
    performance_profile: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class ConfigValidator:
    """Validator for model configurations"""

    def validate_basic_constraints(self, config: ModelConfig) -> ValidationResult:
        """Validate basic configuration constraints"""
        errors = []

        # Check dimensions
        if config.input_dim <= 0:
            errors.append("input_dim must be positive")

        if config.output_dim <= 0:
            errors.append("output_dim must be positive")

        # Check learning rate
        if config.learning_rate <= 0 or config.learning_rate > 1:
            errors.append("learning_rate must be between 0 and 1")

        # Check dropout
        if config.dropout < 0 or config.dropout >= 1:
            errors.append("dropout must be between 0 and 1 (exclusive)")

        # Check batch size
        if config.batch_size <= 0:
            errors.append("batch_size must be positive")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def validate_transductive_constraints(self, config: TransductiveModelConfig) -> ValidationResult:
        """Validate transductive-specific constraints"""
        errors = []

        # Check basic constraints first
        basic_result = self.validate_basic_constraints(config)
        errors.extend(basic_result.errors)

        # Check num_nodes
        if hasattr(config, 'num_nodes') and config.num_nodes <= 0:
            errors.append("num_nodes must be positive")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def validate_hardware_compatibility(self, config: ModelConfig,
                                      hardware_spec: Dict[str, Any]) -> ValidationResult:
        """Validate configuration against hardware constraints"""
        errors = []

        memory_gb = hardware_spec.get('memory_gb', 0)
        device = hardware_spec.get('device', 'cpu')

        # Estimate memory requirements (simplified)
        estimated_params = config.input_dim * sum(config.hidden_dims) + sum(config.hidden_dims) * config.output_dim
        estimated_memory_gb = (estimated_params * config.batch_size * 4) / (1024**3)  # 4 bytes per float32

        if estimated_memory_gb > memory_gb:
            errors.append(f"Estimated memory requirement ({estimated_memory_gb:.2f}GB) exceeds available memory ({memory_gb}GB)")

        # Check GPU requirements
        if device == 'cpu' and config.batch_size > 512:
            errors.append("Large batch sizes may be slow on CPU")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)


class ConfigManager:
    """Manager for model configurations"""

    def __init__(self, storage_path: str = "/tmp/config_storage"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.configs: Dict[str, ModelConfig] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.templates: Dict[str, ConfigTemplate] = {}
        self.presets: Dict[str, ConfigPreset] = {}
        self.validator = ConfigValidator()

    def save_config(self, config: ModelConfig, name: str,
                   description: str = "") -> str:
        """Save a configuration"""
        config_id = str(uuid.uuid4())

        self.configs[config_id] = config
        self.metadata[config_id] = {
            'name': name,
            'description': description,
            'created_at': datetime.now(),
            'config_type': type(config).__name__
        }

        return config_id

    def get_config(self, config_id: str) -> Optional[ModelConfig]:
        """Get configuration by ID"""
        return self.configs.get(config_id)

    def validate_config(self, config: ModelConfig) -> ValidationResult:
        """Validate a configuration"""
        if isinstance(config, TransductiveModelConfig):
            return self.validator.validate_transductive_constraints(config)
        else:
            return self.validator.validate_basic_constraints(config)

    def register_template(self, template: ConfigTemplate) -> bool:
        """Register a configuration template"""
        try:
            self.templates[template.template_name] = template
            return True
        except Exception:
            return False

    def create_config_from_template(self, template_name: str,
                                  overrides: Dict[str, Any] = None) -> Optional[ModelConfig]:
        """Create configuration from template"""
        template = self.templates.get(template_name)
        if not template:
            return None

        overrides = overrides or {}
        return template.generate_config(**overrides)

    def register_preset(self, preset: ConfigPreset) -> bool:
        """Register a configuration preset"""
        try:
            self.presets[preset.preset_name] = preset
            return True
        except Exception:
            return False

    def list_presets(self) -> List[ConfigPreset]:
        """List all registered presets"""
        return list(self.presets.values())

    def get_preset_config(self, preset_name: str) -> Optional[ModelConfig]:
        """Get configuration from preset"""
        preset = self.presets.get(preset_name)
        return preset.config if preset else None

    def compare_configs(self, config1: ModelConfig, config2: ModelConfig) -> Dict[str, Any]:
        """Compare two configurations"""
        config1_dict = asdict(config1)
        config2_dict = asdict(config2)

        differences = {}
        all_keys = set(config1_dict.keys()) | set(config2_dict.keys())

        for key in all_keys:
            val1 = config1_dict.get(key)
            val2 = config2_dict.get(key)

            if val1 != val2:
                differences[key] = {
                    'config1': val1,
                    'config2': val2
                }

        return {
            'differences': differences,
            'similarity_score': 1.0 - len(differences) / len(all_keys)
        }

    def export_config(self, config_id: str, export_path: str) -> bool:
        """Export configuration to file"""
        try:
            config = self.configs.get(config_id)
            metadata = self.metadata.get(config_id)

            if not config or not metadata:
                return False

            export_data = {
                'config': asdict(config),
                'metadata': {
                    'name': metadata['name'],
                    'description': metadata['description'],
                    'created_at': metadata['created_at'].isoformat(),
                    'config_type': metadata['config_type']
                }
            }

            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            return True
        except Exception:
            return False

    def import_config(self, import_path: str, name: str) -> Optional[str]:
        """Import configuration from file"""
        try:
            if not os.path.exists(import_path):
                return None

            with open(import_path, 'r') as f:
                import_data = json.load(f)

            config_data = import_data['config']
            metadata = import_data['metadata']

            # Recreate config object based on type
            config_type = metadata['config_type']
            if config_type == 'TransductiveModelConfig':
                config = TransductiveModelConfig(**config_data)
            elif config_type == 'InductiveModelConfig':
                config = InductiveModelConfig(**config_data)
            else:
                config = ModelConfig(**config_data)

            return self.save_config(config, name, metadata.get('description', ''))
        except Exception:
            return None


class HyperparameterTuner:
    """Hyperparameter tuning utilities"""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def suggest_hyperparameters(self, search_space: Dict[str, Dict[str, Any]],
                              n_suggestions: int = 1) -> List[Dict[str, Any]]:
        """Suggest hyperparameter values"""
        suggestions = []

        for _ in range(n_suggestions):
            suggestion = {}

            for param_name, param_config in search_space.items():
                param_type = param_config['type']

                if param_type == 'float':
                    min_val, max_val = param_config['range']
                    scale = param_config.get('scale', 'linear')

                    if scale == 'log':
                        suggestion[param_name] = np.exp(np.random.uniform(
                            np.log(min_val), np.log(max_val)
                        ))
                    else:
                        suggestion[param_name] = np.random.uniform(min_val, max_val)

                elif param_type == 'int':
                    min_val, max_val = param_config['range']
                    suggestion[param_name] = np.random.randint(min_val, max_val + 1)

                elif param_type == 'choice':
                    choices = param_config['choices']
                    selected_idx = np.random.randint(0, len(choices))
                    selected = choices[selected_idx]
                    suggestion[param_name] = selected.tolist() if isinstance(selected, np.ndarray) else selected

            suggestions.append(suggestion)

        return suggestions

    def report_performance(self, hyperparameters: Dict[str, Any], performance: float):
        """Report performance for given hyperparameters"""
        self.history.append({
            'hyperparameters': hyperparameters,
            'performance': performance,
            'timestamp': datetime.now()
        })

    def get_best_suggestions(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get best performing hyperparameter suggestions"""
        if not self.history:
            return []

        sorted_history = sorted(self.history, key=lambda x: x['performance'], reverse=True)
        return sorted_history[:top_k]


class ConfigurationSchema:
    """Schema definition and validation for configurations"""

    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}

    def register_schema(self, schema_name: str, schema_definition: Dict[str, Any]):
        """Register a configuration schema"""
        self.schemas[schema_name] = schema_definition

    def validate_config_dict(self, schema_name: str, config_dict: Dict[str, Any]) -> ValidationResult:
        """Validate configuration dictionary against schema"""
        schema = self.schemas.get(schema_name)
        if not schema:
            return ValidationResult(is_valid=False, errors=[f"Schema '{schema_name}' not found"])

        errors = []

        # Check required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in config_dict:
                errors.append(f"Required field '{field}' is missing")

        # Check field types and constraints
        properties = schema.get('properties', {})
        for field, value in config_dict.items():
            if field in properties:
                field_schema = properties[field]
                field_errors = self._validate_field(field, value, field_schema)
                errors.extend(field_errors)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _validate_field(self, field_name: str, value: Any, field_schema: Dict[str, Any]) -> List[str]:
        """Validate individual field against its schema"""
        errors = []

        field_type = field_schema.get('type')
        if field_type == 'integer' and not isinstance(value, int):
            errors.append(f"Field '{field_name}' must be an integer")
        elif field_type == 'number' and not isinstance(value, (int, float)):
            errors.append(f"Field '{field_name}' must be a number")
        elif field_type == 'string' and not isinstance(value, str):
            errors.append(f"Field '{field_name}' must be a string")

        # Check minimum/maximum constraints
        if 'minimum' in field_schema and value < field_schema['minimum']:
            errors.append(f"Field '{field_name}' must be >= {field_schema['minimum']}")
        if 'maximum' in field_schema and value > field_schema['maximum']:
            errors.append(f"Field '{field_name}' must be <= {field_schema['maximum']}")

        return errors