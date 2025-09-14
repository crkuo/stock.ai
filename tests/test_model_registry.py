import pytest
import torch
from typing import Dict, Any, List, Optional
from src.models.base_gnn_model import BaseGNNModel, ModelConfig
from src.models.transductive_model import TransductiveGNNModel, TransductiveModelConfig
from src.models.inductive_model import InductiveGNNModel, InductiveModelConfig
from src.models.model_registry import ModelRegistry, ModelFactory, RegisteredModel


class TestModelRegistry:
    def test_should_register_and_retrieve_models(self):
        """Test basic model registration and retrieval"""
        registry = ModelRegistry()

        # Register a transductive model
        transductive_config = TransductiveModelConfig(
            input_dim=16,
            output_dim=1,
            num_nodes=50
        )

        registry.register_model(
            model_name="stock_gcn_v1",
            model_class=TransductiveGNNModel,
            model_config=transductive_config,
            description="Basic GCN for stock analysis",
            tags=["transductive", "gcn", "stocks"]
        )

        # Retrieve registered model info
        model_info = registry.get_model_info("stock_gcn_v1")

        assert model_info is not None
        assert model_info.model_name == "stock_gcn_v1"
        assert model_info.model_class == TransductiveGNNModel
        assert model_info.learning_paradigm == "transductive"
        assert "gcn" in model_info.tags

    def test_should_list_registered_models(self):
        """Test listing all registered models"""
        registry = ModelRegistry()

        # Register multiple models
        registry.register_model(
            "gcn_model",
            TransductiveGNNModel,
            TransductiveModelConfig(input_dim=8, output_dim=1),
            "GCN model"
        )

        registry.register_model(
            "graphsage_model",
            InductiveGNNModel,
            InductiveModelConfig(input_dim=8, output_dim=1),
            "GraphSAGE model"
        )

        # List all models
        models = registry.list_models()

        assert len(models) == 2
        model_names = [model.model_name for model in models]
        assert "gcn_model" in model_names
        assert "graphsage_model" in model_names

    def test_should_filter_models_by_criteria(self):
        """Test filtering models by various criteria"""
        registry = ModelRegistry()

        # Register models with different characteristics
        registry.register_model(
            "transductive_model",
            TransductiveGNNModel,
            TransductiveModelConfig(input_dim=10, output_dim=1),
            "Transductive model",
            tags=["transductive", "semi_supervised"]
        )

        registry.register_model(
            "inductive_model",
            InductiveGNNModel,
            InductiveModelConfig(input_dim=10, output_dim=1),
            "Inductive model",
            tags=["inductive", "scalable"]
        )

        # Filter by learning paradigm
        transductive_models = registry.filter_models(learning_paradigm="transductive")
        assert len(transductive_models) == 1
        assert transductive_models[0].model_name == "transductive_model"

        # Filter by tags
        scalable_models = registry.filter_models(tags=["scalable"])
        assert len(scalable_models) == 1
        assert scalable_models[0].model_name == "inductive_model"

    def test_should_validate_model_registration(self):
        """Test model registration validation"""
        registry = ModelRegistry()

        # Valid registration should succeed
        valid_result = registry.register_model(
            "valid_model",
            TransductiveGNNModel,
            TransductiveModelConfig(input_dim=5, output_dim=1),
            "Valid model"
        )
        assert valid_result is True

        # Duplicate name should fail
        duplicate_result = registry.register_model(
            "valid_model",  # Same name
            InductiveGNNModel,
            InductiveModelConfig(input_dim=5, output_dim=1),
            "Duplicate model"
        )
        assert duplicate_result is False

    def test_should_support_model_versioning(self):
        """Test model versioning capabilities"""
        registry = ModelRegistry()

        # Register version 1
        registry.register_model(
            "stock_predictor",
            TransductiveGNNModel,
            TransductiveModelConfig(input_dim=12, output_dim=1),
            "Stock predictor v1",
            version="1.0.0"
        )

        # Register version 2
        registry.register_model(
            "stock_predictor",
            TransductiveGNNModel,
            TransductiveModelConfig(input_dim=12, output_dim=2),  # Different config
            "Stock predictor v2",
            version="2.0.0"
        )

        # Get specific version
        v1_info = registry.get_model_info("stock_predictor", version="1.0.0")
        v2_info = registry.get_model_info("stock_predictor", version="2.0.0")

        assert v1_info is not None
        assert v2_info is not None
        assert v1_info.config.output_dim == 1
        assert v2_info.config.output_dim == 2

        # Get latest version (default)
        latest_info = registry.get_model_info("stock_predictor")
        assert latest_info.version == "2.0.0"


class TestModelFactory:
    def test_should_create_model_from_registry(self):
        """Test creating model instances from registry"""
        registry = ModelRegistry()
        factory = ModelFactory(registry)

        # Register a model
        registry.register_model(
            "test_gcn",
            TransductiveGNNModel,
            TransductiveModelConfig(input_dim=6, output_dim=1, num_nodes=25),
            "Test GCN model"
        )

        # Create model instance
        model = factory.create_model("test_gcn")

        assert model is not None
        assert isinstance(model, TransductiveGNNModel)
        assert model.config.input_dim == 6
        assert model.config.num_nodes == 25

    def test_should_create_model_with_custom_config(self):
        """Test creating model with custom configuration"""
        registry = ModelRegistry()
        factory = ModelFactory(registry)

        # Register base model
        registry.register_model(
            "base_model",
            InductiveGNNModel,
            InductiveModelConfig(input_dim=8, output_dim=1),
            "Base model"
        )

        # Create with custom config
        custom_config = InductiveModelConfig(input_dim=8, output_dim=2, learning_rate=0.001)
        model = factory.create_model("base_model", custom_config=custom_config)

        assert model is not None
        assert model.config.output_dim == 2
        assert model.config.learning_rate == 0.001

    def test_should_support_model_creation_with_parameters(self):
        """Test creating models with additional parameters"""
        registry = ModelRegistry()
        factory = ModelFactory(registry)

        # Register model
        registry.register_model(
            "parameterized_model",
            TransductiveGNNModel,
            TransductiveModelConfig(input_dim=10, output_dim=1),
            "Parameterized model"
        )

        # Create with additional parameters
        model = factory.create_model(
            "parameterized_model",
            initialize_weights=True,
            device="cpu"
        )

        assert model is not None
        assert isinstance(model, TransductiveGNNModel)

    def test_should_handle_model_creation_errors(self):
        """Test error handling in model creation"""
        registry = ModelRegistry()
        factory = ModelFactory(registry)

        # Try to create non-existent model
        model = factory.create_model("non_existent_model")
        assert model is None

        # Register model with invalid config
        registry.register_model(
            "invalid_model",
            TransductiveGNNModel,
            None,  # Invalid config
            "Invalid model"
        )

        model = factory.create_model("invalid_model")
        assert model is None

    def test_should_support_batch_model_creation(self):
        """Test creating multiple models in batch"""
        registry = ModelRegistry()
        factory = ModelFactory(registry)

        # Register multiple models
        registry.register_model(
            "model_a",
            TransductiveGNNModel,
            TransductiveModelConfig(input_dim=4, output_dim=1),
            "Model A"
        )

        registry.register_model(
            "model_b",
            InductiveGNNModel,
            InductiveModelConfig(input_dim=4, output_dim=1),
            "Model B"
        )

        # Create batch
        model_names = ["model_a", "model_b"]
        models = factory.create_models_batch(model_names)

        assert len(models) == 2
        assert "model_a" in models
        assert "model_b" in models
        assert isinstance(models["model_a"], TransductiveGNNModel)
        assert isinstance(models["model_b"], InductiveGNNModel)

    def test_should_support_model_comparison_creation(self):
        """Test creating models for comparison purposes"""
        registry = ModelRegistry()
        factory = ModelFactory(registry)

        # Register models with same interface but different implementations
        registry.register_model(
            "gcn_baseline",
            TransductiveGNNModel,
            TransductiveModelConfig(input_dim=8, output_dim=1, num_nodes=30),
            "GCN baseline"
        )

        registry.register_model(
            "graphsage_candidate",
            InductiveGNNModel,
            InductiveModelConfig(input_dim=8, output_dim=1),
            "GraphSAGE candidate"
        )

        # Create for comparison
        comparison_models = factory.create_comparison_models([
            "gcn_baseline",
            "graphsage_candidate"
        ])

        assert len(comparison_models) == 2
        assert all(model is not None for model in comparison_models.values())


class TestRegisteredModel:
    def test_should_create_registered_model_metadata(self):
        """Test creating registered model metadata"""
        config = TransductiveModelConfig(input_dim=12, output_dim=1)

        registered_model = RegisteredModel(
            model_name="test_model",
            model_class=TransductiveGNNModel,
            config=config,
            description="Test model for unit testing",
            tags=["test", "transductive"],
            version="1.0.0",
            learning_paradigm="transductive"
        )

        assert registered_model.model_name == "test_model"
        assert registered_model.model_class == TransductiveGNNModel
        assert registered_model.config == config
        assert registered_model.learning_paradigm == "transductive"
        assert "test" in registered_model.tags

    def test_should_validate_registered_model_compatibility(self):
        """Test registered model compatibility validation"""
        registered_model = RegisteredModel(
            model_name="compatibility_test",
            model_class=InductiveGNNModel,
            config=InductiveModelConfig(input_dim=6, output_dim=1),
            description="Compatibility test model",
            learning_paradigm="inductive"
        )

        # Test input compatibility
        test_data = {
            'node_features': torch.randn(20, 6),  # Correct input_dim
            'requires_inductive': True
        }

        compatibility = registered_model.check_compatibility(test_data)
        assert compatibility is True

        # Test incompatible input
        incompatible_data = {
            'node_features': torch.randn(20, 10),  # Wrong input_dim
            'requires_inductive': True
        }

        compatibility = registered_model.check_compatibility(incompatible_data)
        assert compatibility is False

    def test_should_provide_model_metadata_summary(self):
        """Test getting model metadata summary"""
        registered_model = RegisteredModel(
            model_name="summary_test",
            model_class=TransductiveGNNModel,
            config=TransductiveModelConfig(input_dim=8, output_dim=2, num_nodes=50),
            description="Summary test model",
            tags=["summary", "test"],
            version="2.1.0",
            learning_paradigm="transductive"
        )

        summary = registered_model.get_summary()

        assert summary is not None
        assert summary['model_name'] == "summary_test"
        assert summary['learning_paradigm'] == "transductive"
        assert summary['input_dim'] == 8
        assert summary['output_dim'] == 2
        assert summary['version'] == "2.1.0"