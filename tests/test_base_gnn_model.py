import pytest
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.models.base_gnn_model import BaseGNNModel, ModelConfig, ModelMetadata


class TestBaseGNNModel:
    def test_should_create_base_model_with_config(self):
        """Test creating a base GNN model with configuration"""
        config = ModelConfig(
            input_dim=32,
            hidden_dims=[64, 64, 32],
            output_dim=1,
            num_layers=3,
            dropout=0.2,
            learning_rate=0.001
        )

        model = BaseGNNModel(config=config)

        assert model is not None
        assert model.config == config
        assert model.input_dim == 32
        assert model.output_dim == 1
        assert model.num_layers == 3

    def test_should_handle_model_metadata(self):
        """Test model metadata management"""
        config = ModelConfig(input_dim=16, output_dim=1)
        model = BaseGNNModel(config=config)

        metadata = ModelMetadata(
            model_name="test_gnn_v1",
            version="1.0.0",
            created_at=datetime.now(),
            description="Test GNN model for stock analysis",
            tags=["test", "gnn", "stocks"]
        )

        model.set_metadata(metadata)

        retrieved_metadata = model.get_metadata()
        assert retrieved_metadata is not None
        assert retrieved_metadata.model_name == "test_gnn_v1"
        assert retrieved_metadata.version == "1.0.0"
        assert "test" in retrieved_metadata.tags

    def test_should_validate_input_dimensions(self):
        """Test input validation for model training"""
        config = ModelConfig(input_dim=10, output_dim=1)
        model = BaseGNNModel(config=config)

        # Valid input
        node_features = torch.randn(5, 10)  # 5 nodes, 10 features each
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_weights = torch.randn(3)

        validation_result = model.validate_input(
            node_features=node_features,
            edge_index=edge_index,
            edge_weights=edge_weights
        )

        assert validation_result is True

        # Invalid input dimensions
        invalid_features = torch.randn(5, 15)  # Wrong feature dimension
        validation_result = model.validate_input(
            node_features=invalid_features,
            edge_index=edge_index,
            edge_weights=edge_weights
        )

        assert validation_result is False

    def test_should_implement_training_interface(self):
        """Test training interface methods"""
        config = ModelConfig(input_dim=8, output_dim=1)
        model = BaseGNNModel(config=config)

        # Prepare training data
        train_data = {
            'node_features': torch.randn(10, 8),
            'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
            'edge_weights': torch.randn(4),
            'targets': torch.randn(10, 1)
        }

        val_data = {
            'node_features': torch.randn(5, 8),
            'edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            'edge_weights': torch.randn(2),
            'targets': torch.randn(5, 1)
        }

        # Test training
        training_result = model.train_model(
            train_data=train_data,
            validation_data=val_data,
            epochs=5
        )

        assert training_result is not None
        assert 'train_loss' in training_result
        assert 'val_loss' in training_result
        assert 'epochs_completed' in training_result

    def test_should_implement_prediction_interface(self):
        """Test prediction interface methods"""
        config = ModelConfig(input_dim=6, output_dim=1)
        model = BaseGNNModel(config=config)

        # Prepare prediction data
        pred_data = {
            'node_features': torch.randn(8, 6),
            'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            'edge_weights': torch.randn(3)
        }

        # Test prediction
        predictions = model.predict(pred_data)

        assert predictions is not None
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape[0] == 8  # One prediction per node

    def test_should_handle_model_persistence(self):
        """Test model saving and loading"""
        config = ModelConfig(input_dim=4, output_dim=1)
        model = BaseGNNModel(config=config)

        # Test saving model
        save_path = "/tmp/test_gnn_model.pt"
        save_result = model.save_model(save_path)
        assert save_result is True

        # Test loading model
        loaded_model = BaseGNNModel.load_model(save_path)
        assert loaded_model is not None
        assert loaded_model.config.input_dim == 4
        assert loaded_model.config.output_dim == 1

    def test_should_implement_evaluation_metrics(self):
        """Test model evaluation capabilities"""
        config = ModelConfig(input_dim=5, output_dim=1)
        model = BaseGNNModel(config=config)

        # Prepare test data
        test_data = {
            'node_features': torch.randn(6, 5),
            'edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            'edge_weights': torch.randn(2),
            'targets': torch.randn(6, 1)
        }

        # Test evaluation
        eval_results = model.evaluate(test_data)

        assert eval_results is not None
        assert 'mse' in eval_results
        assert 'mae' in eval_results
        assert 'r2_score' in eval_results

    def test_should_support_model_configuration_updates(self):
        """Test dynamic configuration updates"""
        config = ModelConfig(input_dim=3, output_dim=1, learning_rate=0.01)
        model = BaseGNNModel(config=config)

        # Update configuration
        new_config = ModelConfig(input_dim=3, output_dim=1, learning_rate=0.001)
        update_result = model.update_config(new_config)

        assert update_result is True
        assert model.config.learning_rate == 0.001

    def test_should_provide_model_summary(self):
        """Test model summary and information"""
        config = ModelConfig(
            input_dim=12,
            hidden_dims=[24, 16],
            output_dim=1,
            num_layers=2
        )
        model = BaseGNNModel(config=config)

        summary = model.get_model_summary()

        assert summary is not None
        assert 'total_parameters' in summary
        assert 'layer_info' in summary
        assert 'config' in summary
        assert summary['total_parameters'] > 0

    def test_should_handle_model_states(self):
        """Test model state management"""
        config = ModelConfig(input_dim=7, output_dim=1)
        model = BaseGNNModel(config=config)

        # Test initial state
        assert model.is_trained() is False

        # Mock training to change state
        model._set_trained_state(True)
        assert model.is_trained() is True

        # Test model reset
        model.reset_model()
        assert model.is_trained() is False