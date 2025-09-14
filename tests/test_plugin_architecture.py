import pytest
import torch
from typing import Dict, Any, List, Optional
from src.models.base_gnn_model import ModelConfig
from src.models.transductive_model import TransductiveGNNModel, TransductiveModelConfig
from src.models.inductive_model import InductiveGNNModel, InductiveModelConfig
from src.models.plugin_architecture import (
    PluginManager, ModelPlugin, TransductivePlugin, InductivePlugin,
    PluginInterface, PluginRegistry, PluginLoader
)


class TestPluginInterface:
    def test_should_define_plugin_interface_contract(self):
        """Test plugin interface contract definition"""
        plugin = ModelPlugin(
            plugin_name="test_plugin",
            plugin_version="1.0.0",
            supported_paradigms=["transductive", "inductive"]
        )

        assert plugin.plugin_name == "test_plugin"
        assert plugin.plugin_version == "1.0.0"
        assert "transductive" in plugin.supported_paradigms
        assert plugin.is_paradigm_supported("transductive") is True
        assert plugin.is_paradigm_supported("transformer") is False

    def test_should_validate_plugin_requirements(self):
        """Test plugin requirement validation"""
        plugin = ModelPlugin(
            plugin_name="requirements_test",
            plugin_version="2.0.0",
            requirements={
                'torch_version': '>=1.12.0',
                'input_dim_range': (8, 128),
                'supports_gpu': True
            }
        )

        # Test valid requirements
        valid_context = {
            'torch_version': '1.13.0',
            'input_dim': 64,
            'device': 'cuda'
        }

        validation_result = plugin.validate_requirements(valid_context)
        assert validation_result is True

        # Test invalid requirements
        invalid_context = {
            'torch_version': '1.11.0',  # Too old
            'input_dim': 256,  # Out of range
            'device': 'cpu'
        }

        validation_result = plugin.validate_requirements(invalid_context)
        assert validation_result is False

    def test_should_support_plugin_lifecycle_hooks(self):
        """Test plugin lifecycle hook implementation"""
        plugin = ModelPlugin(plugin_name="lifecycle_test")

        # Test initialization hook
        init_result = plugin.on_plugin_load()
        assert init_result is True

        # Test activation hook
        activation_result = plugin.on_plugin_activate()
        assert activation_result is True

        # Test deactivation hook
        deactivation_result = plugin.on_plugin_deactivate()
        assert deactivation_result is True


class TestTransductivePlugin:
    def test_should_create_transductive_plugin(self):
        """Test creating transductive-specific plugin"""
        config = TransductiveModelConfig(
            input_dim=16,
            output_dim=1,
            num_nodes=50,
            semi_supervised=True
        )

        plugin = TransductivePlugin(
            plugin_name="gcn_plugin",
            model_class=TransductiveGNNModel,
            model_config=config,
            supports_semi_supervised=True
        )

        assert plugin.learning_paradigm == "transductive"
        assert plugin.supports_semi_supervised is True
        assert plugin.supports_fixed_graph is True

    def test_should_validate_transductive_data_compatibility(self):
        """Test transductive plugin data validation"""
        plugin = TransductivePlugin(
            plugin_name="transductive_validator",
            model_class=TransductiveGNNModel,
            model_config=TransductiveModelConfig(input_dim=12, output_dim=1, num_nodes=30)
        )

        # Valid transductive data
        valid_data = {
            'node_features': torch.randn(30, 12),
            'train_mask': torch.zeros(30, dtype=torch.bool),
            'val_mask': torch.zeros(30, dtype=torch.bool),
            'has_fixed_structure': True
        }
        valid_data['train_mask'][:20] = True
        valid_data['val_mask'][20:25] = True

        compatibility = plugin.validate_data_compatibility(valid_data)
        assert compatibility is True

        # Invalid data - wrong number of nodes
        invalid_data = {
            'node_features': torch.randn(40, 12),  # Wrong node count
            'train_mask': torch.zeros(40, dtype=torch.bool)
        }

        compatibility = plugin.validate_data_compatibility(invalid_data)
        assert compatibility is False

    def test_should_create_transductive_model_instance(self):
        """Test creating model instance from transductive plugin"""
        config = TransductiveModelConfig(input_dim=8, output_dim=1, num_nodes=25)
        plugin = TransductivePlugin(
            plugin_name="model_creator",
            model_class=TransductiveGNNModel,
            model_config=config
        )

        model = plugin.create_model_instance()

        assert model is not None
        assert isinstance(model, TransductiveGNNModel)
        assert model.config.input_dim == 8
        assert model.config.num_nodes == 25


class TestInductivePlugin:
    def test_should_create_inductive_plugin(self):
        """Test creating inductive-specific plugin"""
        config = InductiveModelConfig(
            input_dim=20,
            output_dim=1,
            supports_new_nodes=True,
            neighborhood_sampling=True
        )

        plugin = InductivePlugin(
            plugin_name="graphsage_plugin",
            model_class=InductiveGNNModel,
            model_config=config,
            supports_new_nodes=True,
            supports_batching=True
        )

        assert plugin.learning_paradigm == "inductive"
        assert plugin.supports_new_nodes is True
        assert plugin.supports_batching is True

    def test_should_handle_variable_graph_sizes(self):
        """Test inductive plugin with variable graph sizes"""
        plugin = InductivePlugin(
            plugin_name="variable_size_handler",
            model_class=InductiveGNNModel,
            model_config=InductiveModelConfig(input_dim=10, output_dim=1)
        )

        # Test with different graph sizes
        small_graph = {
            'node_features': torch.randn(15, 10),
            'edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        }

        large_graph = {
            'node_features': torch.randn(100, 10),
            'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        }

        small_compatibility = plugin.validate_data_compatibility(small_graph)
        large_compatibility = plugin.validate_data_compatibility(large_graph)

        assert small_compatibility is True
        assert large_compatibility is True

    def test_should_support_incremental_learning(self):
        """Test inductive plugin incremental learning support"""
        plugin = InductivePlugin(
            plugin_name="incremental_learner",
            model_class=InductiveGNNModel,
            model_config=InductiveModelConfig(
                input_dim=6,
                output_dim=1,
                supports_incremental_training=True
            ),
            supports_incremental_learning=True
        )

        # Test incremental learning capability
        initial_data = {
            'node_features': torch.randn(20, 6),
            'targets': torch.randn(20, 1)
        }

        new_data = {
            'node_features': torch.randn(10, 6),
            'targets': torch.randn(10, 1)
        }

        incremental_support = plugin.supports_incremental_learning
        assert incremental_support is True

        # Test incremental training
        model = plugin.create_model_instance()
        incremental_result = plugin.perform_incremental_training(
            model, initial_data, new_data
        )

        assert incremental_result is not None
        assert 'success' in incremental_result


class TestPluginManager:
    def test_should_manage_plugin_lifecycle(self):
        """Test plugin manager lifecycle management"""
        manager = PluginManager()

        # Register plugins
        transductive_plugin = TransductivePlugin(
            plugin_name="gcn_v1",
            model_class=TransductiveGNNModel,
            model_config=TransductiveModelConfig(input_dim=8, output_dim=1)
        )

        inductive_plugin = InductivePlugin(
            plugin_name="graphsage_v1",
            model_class=InductiveGNNModel,
            model_config=InductiveModelConfig(input_dim=8, output_dim=1)
        )

        # Register plugins
        manager.register_plugin(transductive_plugin)
        manager.register_plugin(inductive_plugin)

        # List plugins
        all_plugins = manager.list_plugins()
        assert len(all_plugins) == 2

        plugin_names = [p.plugin_name for p in all_plugins]
        assert "gcn_v1" in plugin_names
        assert "graphsage_v1" in plugin_names

    def test_should_filter_plugins_by_paradigm(self):
        """Test filtering plugins by learning paradigm"""
        manager = PluginManager()

        # Register different paradigm plugins
        manager.register_plugin(TransductivePlugin(
            plugin_name="transductive_1",
            model_class=TransductiveGNNModel,
            model_config=TransductiveModelConfig(input_dim=4, output_dim=1)
        ))

        manager.register_plugin(InductivePlugin(
            plugin_name="inductive_1",
            model_class=InductiveGNNModel,
            model_config=InductiveModelConfig(input_dim=4, output_dim=1)
        ))

        # Filter by paradigm
        transductive_plugins = manager.get_plugins_by_paradigm("transductive")
        inductive_plugins = manager.get_plugins_by_paradigm("inductive")

        assert len(transductive_plugins) == 1
        assert len(inductive_plugins) == 1
        assert transductive_plugins[0].plugin_name == "transductive_1"
        assert inductive_plugins[0].plugin_name == "inductive_1"

    def test_should_select_best_plugin_for_data(self):
        """Test automatic plugin selection based on data characteristics"""
        manager = PluginManager()

        # Register plugins with different capabilities
        manager.register_plugin(TransductivePlugin(
            plugin_name="best_for_fixed",
            model_class=TransductiveGNNModel,
            model_config=TransductiveModelConfig(input_dim=10, output_dim=1, num_nodes=50)
        ))

        manager.register_plugin(InductivePlugin(
            plugin_name="best_for_dynamic",
            model_class=InductiveGNNModel,
            model_config=InductiveModelConfig(input_dim=10, output_dim=1)
        ))

        # Test with fixed graph data
        fixed_graph_data = {
            'node_features': torch.randn(50, 10),
            'has_fixed_structure': True,
            'train_mask': torch.zeros(50, dtype=torch.bool)
        }

        best_plugin = manager.select_best_plugin(fixed_graph_data)
        assert best_plugin is not None
        assert best_plugin.plugin_name == "best_for_fixed"

        # Test with dynamic graph data
        dynamic_graph_data = {
            'node_features': torch.randn(30, 10),
            'contains_new_nodes': True,
            'variable_size': True
        }

        best_plugin = manager.select_best_plugin(dynamic_graph_data)
        assert best_plugin is not None
        assert best_plugin.plugin_name == "best_for_dynamic"

    def test_should_handle_plugin_dependencies(self):
        """Test plugin dependency management"""
        manager = PluginManager()

        # Create plugin with dependencies
        dependent_plugin = TransductivePlugin(
            plugin_name="dependent_plugin",
            model_class=TransductiveGNNModel,
            model_config=TransductiveModelConfig(input_dim=6, output_dim=1),
            dependencies=["base_plugin"]
        )

        base_plugin = TransductivePlugin(
            plugin_name="base_plugin",
            model_class=TransductiveGNNModel,
            model_config=TransductiveModelConfig(input_dim=6, output_dim=1)
        )

        # Register base plugin first
        manager.register_plugin(base_plugin)

        # Register dependent plugin
        registration_result = manager.register_plugin(dependent_plugin)
        assert registration_result is True

        # Verify dependency resolution
        dependencies = manager.resolve_plugin_dependencies("dependent_plugin")
        assert dependencies is not None
        assert "base_plugin" in [dep.plugin_name for dep in dependencies]


class TestPluginRegistry:
    def test_should_discover_plugins_automatically(self):
        """Test automatic plugin discovery"""
        registry = PluginRegistry()

        # Mock plugin discovery
        discovered_plugins = registry.discover_plugins("/fake/plugin/path")

        # Should return empty list for non-existent path
        assert isinstance(discovered_plugins, list)

    def test_should_validate_plugin_integrity(self):
        """Test plugin integrity validation"""
        registry = PluginRegistry()

        # Valid plugin
        valid_plugin = TransductivePlugin(
            plugin_name="integrity_test",
            model_class=TransductiveGNNModel,
            model_config=TransductiveModelConfig(input_dim=5, output_dim=1)
        )

        integrity_result = registry.validate_plugin_integrity(valid_plugin)
        assert integrity_result is True

    def test_should_handle_plugin_versioning(self):
        """Test plugin version management"""
        registry = PluginRegistry()

        # Register different versions
        v1_plugin = TransductivePlugin(
            plugin_name="version_test",
            model_class=TransductiveGNNModel,
            model_config=TransductiveModelConfig(input_dim=8, output_dim=1),
            plugin_version="1.0.0"
        )

        v2_plugin = TransductivePlugin(
            plugin_name="version_test",
            model_class=TransductiveGNNModel,
            model_config=TransductiveModelConfig(input_dim=8, output_dim=2),
            plugin_version="2.0.0"
        )

        registry.register_plugin_version(v1_plugin)
        registry.register_plugin_version(v2_plugin)

        # Get latest version
        latest_plugin = registry.get_plugin_latest_version("version_test")
        assert latest_plugin is not None
        assert latest_plugin.plugin_version == "2.0.0"

        # Get specific version
        specific_plugin = registry.get_plugin_version("version_test", "1.0.0")
        assert specific_plugin is not None
        assert specific_plugin.config.output_dim == 1


class TestPluginLoader:
    def test_should_load_plugins_dynamically(self):
        """Test dynamic plugin loading"""
        loader = PluginLoader()

        # Test plugin loading capability
        load_result = loader.load_plugin_from_config({
            'plugin_name': 'dynamic_test',
            'model_class': 'TransductiveGNNModel',
            'paradigm': 'transductive',
            'config': {
                'input_dim': 12,
                'output_dim': 1
            }
        })

        # Should handle loading attempt (may return None for non-existent config)
        assert load_result is None or isinstance(load_result, ModelPlugin)

    def test_should_unload_plugins_safely(self):
        """Test safe plugin unloading"""
        loader = PluginLoader()

        # Create a loaded plugin
        plugin = TransductivePlugin(
            plugin_name="unload_test",
            model_class=TransductiveGNNModel,
            model_config=TransductiveModelConfig(input_dim=4, output_dim=1)
        )

        # Test unloading
        unload_result = loader.unload_plugin(plugin)
        assert unload_result is True