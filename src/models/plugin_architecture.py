import torch
from typing import Dict, Any, List, Optional, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from .base_gnn_model import BaseGNNModel, ModelConfig
from .transductive_model import TransductiveGNNModel, TransductiveModelConfig
from .inductive_model import InductiveGNNModel, InductiveModelConfig


@dataclass
class PluginInterface:
    """Base interface for all model plugins"""
    plugin_name: str
    plugin_version: str = "1.0.0"
    description: str = ""
    author: str = ""
    supported_paradigms: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def is_paradigm_supported(self, paradigm: str) -> bool:
        """Check if plugin supports given paradigm"""
        return paradigm in self.supported_paradigms

    def validate_requirements(self, context: Dict[str, Any]) -> bool:
        """Validate plugin requirements against context"""
        if not self.requirements:
            return True

        # Check torch version
        if 'torch_version' in self.requirements:
            required_version = self.requirements['torch_version']
            actual_version = context.get('torch_version', '0.0.0')
            # Simplified version check (in practice, use packaging.version)
            if required_version.startswith('>=') and actual_version < required_version[2:]:
                return False

        # Check input dimension range
        if 'input_dim_range' in self.requirements:
            min_dim, max_dim = self.requirements['input_dim_range']
            actual_dim = context.get('input_dim', 0)
            if not (min_dim <= actual_dim <= max_dim):
                return False

        # Check GPU support
        if self.requirements.get('supports_gpu') and context.get('device') != 'cuda':
            return False

        return True

    def on_plugin_load(self) -> bool:
        """Hook called when plugin is loaded"""
        return True

    def on_plugin_activate(self) -> bool:
        """Hook called when plugin is activated"""
        return True

    def on_plugin_deactivate(self) -> bool:
        """Hook called when plugin is deactivated"""
        return True


class ModelPlugin(PluginInterface):
    """Base model plugin class"""

    def __init__(self, plugin_name: str, plugin_version: str = "1.0.0",
                 supported_paradigms: List[str] = None, **kwargs):
        super().__init__(
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            supported_paradigms=supported_paradigms or [],
            **kwargs
        )
        self.is_loaded = False
        self.is_active = False

    def load(self) -> bool:
        """Load the plugin"""
        if self.on_plugin_load():
            self.is_loaded = True
            return True
        return False

    def activate(self) -> bool:
        """Activate the plugin"""
        if self.is_loaded and self.on_plugin_activate():
            self.is_active = True
            return True
        return False

    def deactivate(self) -> bool:
        """Deactivate the plugin"""
        if self.on_plugin_deactivate():
            self.is_active = False
            return True
        return False


class TransductivePlugin(ModelPlugin):
    """Plugin for transductive learning models"""

    def __init__(self, plugin_name: str, model_class: Type[TransductiveGNNModel],
                 model_config: TransductiveModelConfig, supports_semi_supervised: bool = None,
                 **kwargs):
        # Remove plugin-specific args before passing to parent
        plugin_kwargs = {k: v for k, v in kwargs.items()
                        if k not in ['supports_semi_supervised']}

        super().__init__(
            plugin_name=plugin_name,
            supported_paradigms=["transductive"],
            **plugin_kwargs
        )
        self.model_class = model_class
        self.model_config = model_config
        self.config = model_config  # Alias for compatibility
        self.learning_paradigm = "transductive"
        self.supports_semi_supervised = (supports_semi_supervised
                                       if supports_semi_supervised is not None
                                       else getattr(model_config, 'semi_supervised', False))
        self.supports_fixed_graph = True

    def validate_data_compatibility(self, data: Dict[str, torch.Tensor]) -> bool:
        """Validate data compatibility with transductive model"""
        node_features = data.get('node_features')
        if node_features is None:
            return False

        # Check feature dimension
        if node_features.shape[1] != self.model_config.input_dim:
            return False

        # Check number of nodes for fixed graph models
        if hasattr(self.model_config, 'num_nodes'):
            if node_features.shape[0] != self.model_config.num_nodes:
                return False

        # Check for required masks
        if 'train_mask' not in data and 'val_mask' not in data:
            return False

        return True

    def create_model_instance(self, custom_config: Optional[ModelConfig] = None) -> BaseGNNModel:
        """Create model instance from plugin"""
        config = custom_config if custom_config is not None else self.model_config
        return self.model_class(config)


class InductivePlugin(ModelPlugin):
    """Plugin for inductive learning models"""

    def __init__(self, plugin_name: str, model_class: Type[InductiveGNNModel],
                 model_config: InductiveModelConfig, supports_new_nodes: bool = True,
                 supports_batching: bool = False, supports_incremental_learning: bool = False,
                 **kwargs):
        # Remove plugin-specific args before passing to parent
        plugin_kwargs = {k: v for k, v in kwargs.items()
                        if k not in ['supports_new_nodes', 'supports_batching',
                                   'supports_incremental_learning']}

        super().__init__(
            plugin_name=plugin_name,
            supported_paradigms=["inductive"],
            **plugin_kwargs
        )
        self.model_class = model_class
        self.model_config = model_config
        self.config = model_config  # Alias for compatibility
        self.learning_paradigm = "inductive"
        self.supports_new_nodes = supports_new_nodes
        self.supports_batching = supports_batching
        self.supports_incremental_learning = supports_incremental_learning

    def validate_data_compatibility(self, data: Dict[str, torch.Tensor]) -> bool:
        """Validate data compatibility with inductive model"""
        node_features = data.get('node_features')
        if node_features is None:
            return False

        # Check feature dimension
        if node_features.shape[1] != self.model_config.input_dim:
            return False

        # Inductive models can handle variable graph sizes
        return True

    def create_model_instance(self, custom_config: Optional[ModelConfig] = None) -> BaseGNNModel:
        """Create model instance from plugin"""
        config = custom_config if custom_config is not None else self.model_config
        return self.model_class(config)

    def perform_incremental_training(self, model: BaseGNNModel,
                                   initial_data: Dict[str, torch.Tensor],
                                   new_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform incremental training"""
        if not self.supports_incremental_learning:
            return {'success': False, 'reason': 'Incremental learning not supported'}

        try:
            # Simple incremental training implementation
            if hasattr(model, 'train_incremental'):
                result = model.train_incremental(new_data, epochs=5)
                return {'success': True, 'result': result}
            else:
                return {'success': False, 'reason': 'Model does not support incremental training'}
        except Exception as e:
            return {'success': False, 'error': str(e)}


class PluginManager:
    """Manager for handling model plugins"""

    def __init__(self):
        self.plugins: Dict[str, ModelPlugin] = {}
        self.active_plugins: Dict[str, ModelPlugin] = {}

    def register_plugin(self, plugin: ModelPlugin) -> bool:
        """Register a plugin"""
        try:
            if plugin.plugin_name in self.plugins:
                return False  # Plugin already exists

            # Validate dependencies
            for dep in plugin.dependencies:
                if dep not in self.plugins:
                    return False  # Missing dependency

            self.plugins[plugin.plugin_name] = plugin
            plugin.load()
            return True
        except Exception:
            return False

    def list_plugins(self) -> List[ModelPlugin]:
        """List all registered plugins"""
        return list(self.plugins.values())

    def get_plugin(self, plugin_name: str) -> Optional[ModelPlugin]:
        """Get plugin by name"""
        return self.plugins.get(plugin_name)

    def get_plugins_by_paradigm(self, paradigm: str) -> List[ModelPlugin]:
        """Get plugins by learning paradigm"""
        return [plugin for plugin in self.plugins.values()
                if plugin.is_paradigm_supported(paradigm)]

    def activate_plugin(self, plugin_name: str) -> bool:
        """Activate a plugin"""
        plugin = self.plugins.get(plugin_name)
        if plugin and plugin.activate():
            self.active_plugins[plugin_name] = plugin
            return True
        return False

    def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate a plugin"""
        plugin = self.active_plugins.get(plugin_name)
        if plugin and plugin.deactivate():
            del self.active_plugins[plugin_name]
            return True
        return False

    def select_best_plugin(self, data: Dict[str, torch.Tensor]) -> Optional[ModelPlugin]:
        """Select best plugin for given data"""
        compatible_plugins = []

        for plugin in self.plugins.values():
            if plugin.validate_data_compatibility(data):
                compatible_plugins.append(plugin)

        if not compatible_plugins:
            return None

        # Simple selection logic - prioritize based on data characteristics
        if data.get('has_fixed_structure', False):
            # Prefer transductive for fixed structures
            transductive_plugins = [p for p in compatible_plugins
                                  if p.learning_paradigm == "transductive"]
            if transductive_plugins:
                return transductive_plugins[0]

        if data.get('contains_new_nodes', False) or data.get('variable_size', False):
            # Prefer inductive for dynamic graphs
            inductive_plugins = [p for p in compatible_plugins
                               if p.learning_paradigm == "inductive"]
            if inductive_plugins:
                return inductive_plugins[0]

        # Return first compatible plugin
        return compatible_plugins[0]

    def resolve_plugin_dependencies(self, plugin_name: str) -> Optional[List[ModelPlugin]]:
        """Resolve plugin dependencies"""
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return None

        dependencies = []
        for dep_name in plugin.dependencies:
            dep_plugin = self.plugins.get(dep_name)
            if dep_plugin:
                dependencies.append(dep_plugin)

        return dependencies


class PluginRegistry:
    """Registry for plugin discovery and management"""

    def __init__(self):
        self.plugin_versions: Dict[str, Dict[str, ModelPlugin]] = {}

    def discover_plugins(self, plugin_path: str) -> List[ModelPlugin]:
        """Discover plugins from path"""
        # Mock implementation - in practice would scan filesystem
        return []

    def validate_plugin_integrity(self, plugin: ModelPlugin) -> bool:
        """Validate plugin integrity"""
        try:
            # Basic validation
            if not plugin.plugin_name or not plugin.plugin_version:
                return False

            if hasattr(plugin, 'model_class') and hasattr(plugin, 'model_config'):
                if plugin.model_class is None or plugin.model_config is None:
                    return False

            return True
        except Exception:
            return False

    def register_plugin_version(self, plugin: ModelPlugin) -> bool:
        """Register plugin version"""
        try:
            if plugin.plugin_name not in self.plugin_versions:
                self.plugin_versions[plugin.plugin_name] = {}

            self.plugin_versions[plugin.plugin_name][plugin.plugin_version] = plugin
            return True
        except Exception:
            return False

    def get_plugin_latest_version(self, plugin_name: str) -> Optional[ModelPlugin]:
        """Get latest version of plugin"""
        if plugin_name not in self.plugin_versions:
            return None

        versions = self.plugin_versions[plugin_name]
        latest_version = max(versions.keys())
        return versions[latest_version]

    def get_plugin_version(self, plugin_name: str, version: str) -> Optional[ModelPlugin]:
        """Get specific version of plugin"""
        if plugin_name not in self.plugin_versions:
            return None

        return self.plugin_versions[plugin_name].get(version)


class PluginLoader:
    """Loader for dynamic plugin loading"""

    def __init__(self):
        self.loaded_plugins: Dict[str, ModelPlugin] = {}

    def load_plugin_from_config(self, config: Dict[str, Any]) -> Optional[ModelPlugin]:
        """Load plugin from configuration"""
        try:
            plugin_name = config.get('plugin_name')
            paradigm = config.get('paradigm')

            if not plugin_name or not paradigm:
                return None

            # Mock loading - in practice would dynamically import
            return None
        except Exception:
            return None

    def unload_plugin(self, plugin: ModelPlugin) -> bool:
        """Unload plugin safely"""
        try:
            if plugin.plugin_name in self.loaded_plugins:
                plugin.deactivate()
                del self.loaded_plugins[plugin.plugin_name]
            return True
        except Exception:
            return False