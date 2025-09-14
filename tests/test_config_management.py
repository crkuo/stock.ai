import pytest
import torch
import json
import os
from typing import Dict, Any, List, Optional
from src.models.base_gnn_model import ModelConfig
from src.models.transductive_model import TransductiveModelConfig
from src.models.inductive_model import InductiveModelConfig
from src.models.config_management import (
    ConfigManager, ConfigTemplate, ConfigValidator, ConfigPreset,
    HyperparameterTuner, ConfigurationSchema
)


class TestConfigManager:
    def test_should_create_and_manage_configurations(self):
        """Test creating and managing model configurations"""
        manager = ConfigManager()

        # Create a configuration
        config = TransductiveModelConfig(
            input_dim=16,
            output_dim=1,
            hidden_dims=[32, 16],
            num_nodes=50,
            learning_rate=0.01,
            dropout=0.2
        )

        # Save configuration
        config_id = manager.save_config(
            config=config,
            name="test_gcn_config",
            description="Test GCN configuration for stock analysis"
        )

        assert config_id is not None

        # Retrieve configuration
        retrieved_config = manager.get_config(config_id)
        assert retrieved_config is not None
        assert retrieved_config.input_dim == 16
        assert retrieved_config.output_dim == 1
        assert retrieved_config.learning_rate == 0.01

    def test_should_support_configuration_templates(self):
        """Test configuration templates for common scenarios"""
        manager = ConfigManager()

        # Create template for small graphs
        small_graph_template = ConfigTemplate(
            template_name="small_graph_gcn",
            base_config=TransductiveModelConfig(
                input_dim=8,
                output_dim=1,
                hidden_dims=[16, 8],
                num_nodes=30
            ),
            parameter_ranges={
                'learning_rate': (0.001, 0.1),
                'dropout': (0.1, 0.5),
                'num_nodes': (20, 50)
            },
            description="Template for small stock graphs"
        )

        manager.register_template(small_graph_template)

        # Create config from template
        config = manager.create_config_from_template(
            "small_graph_gcn",
            overrides={'learning_rate': 0.005, 'num_nodes': 40}
        )

        assert config is not None
        assert config.learning_rate == 0.005
        assert config.num_nodes == 40
        assert config.input_dim == 8  # From template

    def test_should_validate_configurations(self):
        """Test configuration validation"""
        manager = ConfigManager()

        # Valid configuration
        valid_config = InductiveModelConfig(
            input_dim=12,
            output_dim=1,
            learning_rate=0.01,
            dropout=0.2
        )

        validation_result = manager.validate_config(valid_config)
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0

        # Invalid configuration
        invalid_config = TransductiveModelConfig(
            input_dim=-5,  # Invalid negative dimension
            output_dim=0,   # Invalid zero output
            learning_rate=2.0,  # Invalid high learning rate
            dropout=1.5     # Invalid high dropout
        )

        validation_result = manager.validate_config(invalid_config)
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0

    def test_should_support_configuration_presets(self):
        """Test predefined configuration presets"""
        manager = ConfigManager()

        # Register preset configurations
        high_performance_preset = ConfigPreset(
            preset_name="high_performance_gcn",
            config=TransductiveModelConfig(
                input_dim=32,
                output_dim=1,
                hidden_dims=[128, 64, 32],
                num_layers=3,
                learning_rate=0.001,
                dropout=0.1
            ),
            use_case="High accuracy stock prediction",
            performance_profile="high_memory_high_accuracy"
        )

        fast_inference_preset = ConfigPreset(
            preset_name="fast_inference_graphsage",
            config=InductiveModelConfig(
                input_dim=16,
                output_dim=1,
                hidden_dims=[32, 16],
                num_layers=2,
                learning_rate=0.01,
                neighborhood_sampling=True,
                max_neighbors=5
            ),
            use_case="Real-time stock prediction",
            performance_profile="low_memory_fast_inference"
        )

        manager.register_preset(high_performance_preset)
        manager.register_preset(fast_inference_preset)

        # Get presets
        presets = manager.list_presets()
        assert len(presets) == 2

        preset_names = [p.preset_name for p in presets]
        assert "high_performance_gcn" in preset_names
        assert "fast_inference_graphsage" in preset_names

        # Get specific preset
        hp_config = manager.get_preset_config("high_performance_gcn")
        assert hp_config is not None
        assert hp_config.input_dim == 32
        assert len(hp_config.hidden_dims) == 3

    def test_should_support_configuration_comparison(self):
        """Test comparing different configurations"""
        manager = ConfigManager()

        config1 = TransductiveModelConfig(
            input_dim=16,
            output_dim=1,
            learning_rate=0.01,
            dropout=0.2
        )

        config2 = TransductiveModelConfig(
            input_dim=16,
            output_dim=1,
            learning_rate=0.005,  # Different
            dropout=0.3           # Different
        )

        comparison = manager.compare_configs(config1, config2)

        assert comparison is not None
        assert 'differences' in comparison
        assert 'learning_rate' in comparison['differences']
        assert 'dropout' in comparison['differences']
        assert comparison['differences']['learning_rate']['config1'] == 0.01
        assert comparison['differences']['learning_rate']['config2'] == 0.005


class TestConfigValidator:
    def test_should_validate_basic_constraints(self):
        """Test basic configuration constraint validation"""
        validator = ConfigValidator()

        # Valid config
        valid_config = ModelConfig(
            input_dim=10,
            output_dim=1,
            learning_rate=0.01,
            dropout=0.2
        )

        result = validator.validate_basic_constraints(valid_config)
        assert result.is_valid is True

        # Invalid config
        invalid_config = ModelConfig(
            input_dim=0,      # Invalid
            output_dim=-1,    # Invalid
            learning_rate=-0.1,  # Invalid
            dropout=1.2       # Invalid
        )

        result = validator.validate_basic_constraints(invalid_config)
        assert result.is_valid is False
        assert len(result.errors) >= 4  # Multiple errors

    def test_should_validate_paradigm_specific_constraints(self):
        """Test paradigm-specific validation"""
        validator = ConfigValidator()

        # Valid transductive config
        transductive_config = TransductiveModelConfig(
            input_dim=8,
            output_dim=1,
            num_nodes=50
        )

        result = validator.validate_transductive_constraints(transductive_config)
        assert result.is_valid is True

        # Invalid transductive config
        invalid_transductive = TransductiveModelConfig(
            input_dim=8,
            output_dim=1,
            num_nodes=0  # Invalid
        )

        result = validator.validate_transductive_constraints(invalid_transductive)
        assert result.is_valid is False

    def test_should_validate_hardware_compatibility(self):
        """Test hardware compatibility validation"""
        validator = ConfigValidator()

        # Config that should work on CPU
        cpu_config = ModelConfig(
            input_dim=8,
            output_dim=1,
            batch_size=16
        )

        hardware_spec = {
            'device': 'cpu',
            'memory_gb': 8,
            'cores': 4
        }

        result = validator.validate_hardware_compatibility(cpu_config, hardware_spec)
        assert result.is_valid is True

        # Config too large for hardware
        large_config = ModelConfig(
            input_dim=1000,
            output_dim=100,
            hidden_dims=[2000, 1000, 500],
            batch_size=1000
        )

        small_hardware = {
            'device': 'cpu',
            'memory_gb': 2,
            'cores': 2
        }

        result = validator.validate_hardware_compatibility(large_config, small_hardware)
        assert result.is_valid is False


class TestConfigTemplate:
    def test_should_create_configuration_template(self):
        """Test creating configuration templates"""
        template = ConfigTemplate(
            template_name="stock_analysis_template",
            base_config=TransductiveModelConfig(
                input_dim=20,
                output_dim=1,
                hidden_dims=[40, 20]
            ),
            parameter_ranges={
                'learning_rate': (0.001, 0.1),
                'dropout': (0.1, 0.5)
            },
            required_parameters=['input_dim', 'output_dim'],
            optional_parameters=['dropout', 'weight_decay']
        )

        assert template.template_name == "stock_analysis_template"
        assert template.base_config.input_dim == 20
        assert 'learning_rate' in template.parameter_ranges

    def test_should_generate_config_from_template(self):
        """Test generating configurations from templates"""
        template = ConfigTemplate(
            template_name="flexible_template",
            base_config=InductiveModelConfig(
                input_dim=16,
                output_dim=1
            ),
            parameter_ranges={
                'learning_rate': (0.001, 0.1),
                'batch_size': (16, 128)
            }
        )

        # Generate with specific values
        config = template.generate_config(
            learning_rate=0.005,
            batch_size=64
        )

        assert config is not None
        assert config.learning_rate == 0.005
        assert config.batch_size == 64
        assert config.input_dim == 16  # From base

    def test_should_validate_template_parameters(self):
        """Test template parameter validation"""
        template = ConfigTemplate(
            template_name="validation_template",
            base_config=ModelConfig(input_dim=8, output_dim=1),
            parameter_ranges={
                'learning_rate': (0.001, 0.1)
            }
        )

        # Valid parameter
        validation = template.validate_parameters({'learning_rate': 0.01})
        assert validation.is_valid is True

        # Invalid parameter
        validation = template.validate_parameters({'learning_rate': 0.5})  # Out of range
        assert validation.is_valid is False


class TestHyperparameterTuner:
    def test_should_suggest_hyperparameter_values(self):
        """Test hyperparameter suggestion"""
        tuner = HyperparameterTuner()

        # Define search space
        search_space = {
            'learning_rate': {'type': 'float', 'range': (0.001, 0.1), 'scale': 'log'},
            'dropout': {'type': 'float', 'range': (0.1, 0.5), 'scale': 'linear'},
            'hidden_dims': {'type': 'choice', 'choices': [[32, 16], [64, 32], [128, 64]]},
            'num_layers': {'type': 'int', 'range': (2, 5)}
        }

        # Get suggestions
        suggestions = tuner.suggest_hyperparameters(search_space, n_suggestions=3)

        assert len(suggestions) == 3
        for suggestion in suggestions:
            assert 'learning_rate' in suggestion
            assert 'dropout' in suggestion
            assert 'hidden_dims' in suggestion
            assert 'num_layers' in suggestion

            # Validate ranges
            assert 0.001 <= suggestion['learning_rate'] <= 0.1
            assert 0.1 <= suggestion['dropout'] <= 0.5
            assert 2 <= suggestion['num_layers'] <= 5

    def test_should_optimize_hyperparameters_with_feedback(self):
        """Test hyperparameter optimization with performance feedback"""
        tuner = HyperparameterTuner()

        search_space = {
            'learning_rate': {'type': 'float', 'range': (0.001, 0.1)},
            'dropout': {'type': 'float', 'range': (0.1, 0.5)}
        }

        # Simulate optimization process
        for i in range(3):
            suggestion = tuner.suggest_hyperparameters(search_space, n_suggestions=1)[0]

            # Simulate performance (higher learning rate = worse performance for demo)
            performance = 1.0 - suggestion['learning_rate'] * 5

            tuner.report_performance(suggestion, performance)

        # Get optimized suggestions
        optimized_suggestions = tuner.get_best_suggestions(top_k=1)

        assert len(optimized_suggestions) == 1
        best_suggestion = optimized_suggestions[0]
        assert 'hyperparameters' in best_suggestion
        assert 'performance' in best_suggestion


class TestConfigurationSchema:
    def test_should_define_configuration_schema(self):
        """Test configuration schema definition"""
        schema = ConfigurationSchema()

        # Add schema for transductive models
        transductive_schema = {
            'type': 'object',
            'properties': {
                'input_dim': {'type': 'integer', 'minimum': 1},
                'output_dim': {'type': 'integer', 'minimum': 1},
                'num_nodes': {'type': 'integer', 'minimum': 1},
                'learning_rate': {'type': 'number', 'minimum': 0.0001, 'maximum': 1.0}
            },
            'required': ['input_dim', 'output_dim', 'num_nodes']
        }

        schema.register_schema('transductive', transductive_schema)

        # Validate against schema
        valid_config_dict = {
            'input_dim': 16,
            'output_dim': 1,
            'num_nodes': 50,
            'learning_rate': 0.01
        }

        validation_result = schema.validate_config_dict('transductive', valid_config_dict)
        assert validation_result.is_valid is True

        # Invalid config
        invalid_config_dict = {
            'input_dim': -1,  # Invalid
            'output_dim': 1
            # Missing required num_nodes
        }

        validation_result = schema.validate_config_dict('transductive', invalid_config_dict)
        assert validation_result.is_valid is False

    def test_should_support_configuration_export_import(self):
        """Test configuration export and import"""
        manager = ConfigManager()

        # Create and save config
        config = TransductiveModelConfig(
            input_dim=12,
            output_dim=1,
            learning_rate=0.005,
            dropout=0.3
        )

        config_id = manager.save_config(config, "export_test")

        # Export to file
        export_path = "/tmp/test_config.json"
        export_result = manager.export_config(config_id, export_path)
        assert export_result is True
        assert os.path.exists(export_path)

        # Import from file
        imported_config_id = manager.import_config(export_path, "imported_config")
        assert imported_config_id is not None

        # Verify imported config matches original
        imported_config = manager.get_config(imported_config_id)
        assert imported_config.input_dim == 12
        assert imported_config.learning_rate == 0.005