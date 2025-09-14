import torch
from typing import Dict, Any, List, Optional, Type, Union
from dataclasses import dataclass
from datetime import datetime
from .base_gnn_model import BaseGNNModel, ModelConfig


@dataclass
class RegisteredModel:
    """Metadata for a registered model"""
    model_name: str
    model_class: Type[BaseGNNModel]
    config: ModelConfig
    description: str
    tags: List[str] = None
    version: str = "1.0.0"
    learning_paradigm: str = "base"
    registered_at: datetime = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.registered_at is None:
            self.registered_at = datetime.now()

        # Auto-detect learning paradigm if not set
        if self.learning_paradigm == "base":
            if hasattr(self.model_class, 'get_learning_paradigm'):
                # Create temporary instance to get paradigm
                temp_instance = self.model_class(self.config)
                self.learning_paradigm = temp_instance.get_learning_paradigm()

    def check_compatibility(self, data: Dict[str, Any]) -> bool:
        """Check if model is compatible with given data"""
        node_features = data.get('node_features')

        if node_features is None:
            return False

        # Check input dimension compatibility
        if node_features.shape[1] != self.config.input_dim:
            return False

        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get model metadata summary"""
        return {
            'model_name': self.model_name,
            'model_class': self.model_class.__name__,
            'learning_paradigm': self.learning_paradigm,
            'input_dim': self.config.input_dim,
            'output_dim': self.config.output_dim,
            'version': self.version,
            'description': self.description,
            'tags': self.tags,
            'registered_at': self.registered_at
        }


class ModelRegistry:
    """Registry for managing and discovering GNN models"""

    def __init__(self):
        self._models: Dict[str, Dict[str, RegisteredModel]] = {}  # name -> version -> model

    def register_model(self, model_name: str, model_class: Type[BaseGNNModel],
                      model_config: ModelConfig, description: str,
                      tags: List[str] = None, version: str = "1.0.0") -> bool:
        """Register a model in the registry"""
        try:
            if model_config is None:
                return False

            registered_model = RegisteredModel(
                model_name=model_name,
                model_class=model_class,
                config=model_config,
                description=description,
                tags=tags or [],
                version=version
            )

            # Initialize model name entry if doesn't exist
            if model_name not in self._models:
                self._models[model_name] = {}

            # Check for duplicate version
            if version in self._models[model_name]:
                return False

            self._models[model_name][version] = registered_model
            return True

        except Exception:
            return False

    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Optional[RegisteredModel]:
        """Get information about a registered model"""
        if model_name not in self._models:
            return None

        model_versions = self._models[model_name]

        if version is None:
            # Get latest version
            latest_version = max(model_versions.keys())
            return model_versions[latest_version]
        else:
            return model_versions.get(version)

    def list_models(self) -> List[RegisteredModel]:
        """List all registered models (latest versions only)"""
        models = []
        for model_name, versions in self._models.items():
            latest_version = max(versions.keys())
            models.append(versions[latest_version])
        return models

    def filter_models(self, learning_paradigm: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     input_dim: Optional[int] = None) -> List[RegisteredModel]:
        """Filter models by criteria"""
        all_models = self.list_models()
        filtered_models = []

        for model in all_models:
            # Filter by learning paradigm
            if learning_paradigm and model.learning_paradigm != learning_paradigm:
                continue

            # Filter by tags
            if tags and not any(tag in model.tags for tag in tags):
                continue

            # Filter by input dimension
            if input_dim and model.config.input_dim != input_dim:
                continue

            filtered_models.append(model)

        return filtered_models


class ModelFactory:
    """Factory for creating model instances from registry"""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def create_model(self, model_name: str, version: Optional[str] = None,
                    custom_config: Optional[ModelConfig] = None,
                    **kwargs) -> Optional[BaseGNNModel]:
        """Create a model instance from registry"""
        try:
            model_info = self.registry.get_model_info(model_name, version)

            if model_info is None:
                return None

            # Use custom config if provided, otherwise use registered config
            config = custom_config if custom_config is not None else model_info.config

            if config is None:
                return None

            # Create model instance
            model = model_info.model_class(config)

            # Apply additional parameters if provided
            if kwargs.get('initialize_weights', False):
                self._initialize_weights(model)

            if 'device' in kwargs:
                model = model.to(kwargs['device'])

            return model

        except Exception:
            return None

    def create_models_batch(self, model_names: List[str],
                          custom_configs: Optional[Dict[str, ModelConfig]] = None) -> Dict[str, BaseGNNModel]:
        """Create multiple models in batch"""
        models = {}

        for model_name in model_names:
            custom_config = custom_configs.get(model_name) if custom_configs else None
            model = self.create_model(model_name, custom_config=custom_config)

            if model is not None:
                models[model_name] = model

        return models

    def create_comparison_models(self, model_names: List[str]) -> Dict[str, BaseGNNModel]:
        """Create models for comparison purposes"""
        return self.create_models_batch(model_names)

    def _initialize_weights(self, model: BaseGNNModel):
        """Initialize model weights"""
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)