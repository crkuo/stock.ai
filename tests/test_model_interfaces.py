import pytest
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.models.base_gnn_model import ModelConfig
from src.models.transductive_model import TransductiveGNNModel, TransductiveModelConfig
from src.models.inductive_model import InductiveGNNModel, InductiveModelConfig


class TestTransductiveGNNModel:
    def test_should_create_transductive_model_with_fixed_graph(self):
        """Test creating transductive model with fixed graph structure"""
        config = TransductiveModelConfig(
            input_dim=16,
            output_dim=1,
            num_nodes=100,
            requires_full_graph=True,
            supports_node_addition=False
        )

        model = TransductiveGNNModel(config=config)

        assert model is not None
        assert model.config.num_nodes == 100
        assert model.config.requires_full_graph is True
        assert model.supports_node_addition() is False

    def test_should_handle_fixed_graph_training_data(self):
        """Test training with fixed graph structure"""
        config = TransductiveModelConfig(
            input_dim=8,
            output_dim=1,
            num_nodes=50
        )
        model = TransductiveGNNModel(config=config)

        # Create fixed graph data
        train_data = {
            'node_features': torch.randn(50, 8),
            'edge_index': torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
            'edge_weights': torch.randn(4),
            'targets': torch.randn(50, 1),
            'train_mask': torch.zeros(50, dtype=torch.bool)
        }
        train_data['train_mask'][:30] = True  # First 30 nodes for training

        val_data = {
            'node_features': train_data['node_features'],
            'edge_index': train_data['edge_index'],
            'edge_weights': train_data['edge_weights'],
            'targets': train_data['targets'],
            'val_mask': torch.zeros(50, dtype=torch.bool)
        }
        val_data['val_mask'][30:40] = True  # Next 10 nodes for validation

        training_result = model.train_transductive(
            train_data=train_data,
            validation_data=val_data,
            epochs=5
        )

        assert training_result is not None
        assert 'train_loss' in training_result
        assert 'val_loss' in training_result

    def test_should_support_semi_supervised_learning(self):
        """Test semi-supervised learning capability"""
        config = TransductiveModelConfig(
            input_dim=12,
            output_dim=2,
            num_nodes=60,
            semi_supervised=True
        )
        model = TransductiveGNNModel(config=config)

        # Create semi-supervised data
        data = {
            'node_features': torch.randn(60, 12),
            'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            'edge_weights': torch.randn(3),
            'targets': torch.randn(60, 2),
            'labeled_mask': torch.zeros(60, dtype=torch.bool)
        }
        data['labeled_mask'][:20] = True  # Only 20 labeled nodes

        training_result = model.train_semi_supervised(data, epochs=3)

        assert training_result is not None
        assert 'labeled_loss' in training_result
        assert 'unlabeled_predictions' in training_result

    def test_should_validate_graph_consistency(self):
        """Test graph structure validation for transductive models"""
        config = TransductiveModelConfig(
            input_dim=6,
            output_dim=1,
            num_nodes=25
        )
        model = TransductiveGNNModel(config=config)

        # Valid graph structure
        valid_data = {
            'node_features': torch.randn(25, 6),
            'edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        }

        validation_result = model.validate_graph_structure(valid_data)
        assert validation_result is True

        # Invalid graph - wrong number of nodes
        invalid_data = {
            'node_features': torch.randn(30, 6),  # Wrong number of nodes
            'edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        }

        validation_result = model.validate_graph_structure(invalid_data)
        assert validation_result is False


class TestInductiveGNNModel:
    def test_should_create_inductive_model_for_new_nodes(self):
        """Test creating inductive model that supports new nodes"""
        config = InductiveModelConfig(
            input_dim=20,
            output_dim=1,
            supports_new_nodes=True,
            neighborhood_sampling=True,
            max_neighbors=10
        )

        model = InductiveGNNModel(config=config)

        assert model is not None
        assert model.config.supports_new_nodes is True
        assert model.config.neighborhood_sampling is True
        assert model.supports_new_nodes() is True

    def test_should_handle_variable_graph_sizes(self):
        """Test training with variable graph sizes"""
        config = InductiveModelConfig(
            input_dim=10,
            output_dim=1,
            supports_new_nodes=True
        )
        model = InductiveGNNModel(config=config)

        # Training data with one graph size
        train_data = {
            'node_features': torch.randn(40, 10),
            'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            'edge_weights': torch.randn(3),
            'targets': torch.randn(40, 1),
            'batch_info': torch.zeros(40, dtype=torch.long)  # Single batch
        }

        # Validation data with different graph size
        val_data = {
            'node_features': torch.randn(25, 10),  # Different size
            'edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            'edge_weights': torch.randn(2),
            'targets': torch.randn(25, 1),
            'batch_info': torch.zeros(25, dtype=torch.long)
        }

        training_result = model.train_inductive(
            train_data=train_data,
            validation_data=val_data,
            epochs=5
        )

        assert training_result is not None
        assert 'train_loss' in training_result
        assert 'val_loss' in training_result

    def test_should_support_neighborhood_sampling(self):
        """Test neighborhood sampling for scalability"""
        config = InductiveModelConfig(
            input_dim=8,
            output_dim=1,
            neighborhood_sampling=True,
            max_neighbors=5,
            sampling_layers=[5, 3]  # 2-layer sampling
        )
        model = InductiveGNNModel(config=config)

        # Large graph data
        data = {
            'node_features': torch.randn(100, 8),
            'edge_index': torch.randint(0, 100, (2, 200)),
            'target_nodes': torch.tensor([0, 10, 20, 30, 40])  # Nodes to predict
        }

        sampled_data = model.sample_neighborhood(
            data=data,
            target_nodes=data['target_nodes']
        )

        assert sampled_data is not None
        assert 'subgraph_nodes' in sampled_data
        assert 'subgraph_edges' in sampled_data
        assert len(sampled_data['subgraph_nodes']) <= 100  # Should be reduced

    def test_should_predict_on_new_unseen_nodes(self):
        """Test prediction on completely new nodes"""
        config = InductiveModelConfig(
            input_dim=6,
            output_dim=1,
            supports_new_nodes=True
        )
        model = InductiveGNNModel(config=config)

        # Mock training to set model as trained
        model._set_trained_state(True)

        # New graph with unseen nodes
        new_data = {
            'node_features': torch.randn(15, 6),  # Completely new nodes
            'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            'edge_weights': torch.randn(3)
        }

        predictions = model.predict_new_nodes(new_data)

        assert predictions is not None
        assert predictions.shape[0] == 15  # One prediction per new node

    def test_should_support_incremental_training(self):
        """Test incremental training with new data"""
        config = InductiveModelConfig(
            input_dim=12,
            output_dim=1,
            supports_incremental_training=True
        )
        model = InductiveGNNModel(config=config)

        # Initial training data
        initial_data = {
            'node_features': torch.randn(30, 12),
            'edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            'targets': torch.randn(30, 1)
        }

        # Train initially
        model.train_inductive(initial_data, initial_data, epochs=2)

        # New incremental data
        new_data = {
            'node_features': torch.randn(20, 12),
            'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            'targets': torch.randn(20, 1)
        }

        # Incremental training
        incremental_result = model.train_incremental(
            new_data=new_data,
            learning_rate_decay=0.9,
            epochs=3
        )

        assert incremental_result is not None
        assert 'incremental_loss' in incremental_result
        assert 'model_updated' in incremental_result

    def test_should_handle_graph_batching(self):
        """Test batching multiple graphs for training"""
        config = InductiveModelConfig(
            input_dim=5,
            output_dim=1,
            batch_multiple_graphs=True
        )
        model = InductiveGNNModel(config=config)

        # Create batch of multiple graphs
        graph_batch = {
            'node_features': torch.randn(50, 5),  # Features for all nodes
            'edge_indices': [
                torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),  # Graph 1
                torch.tensor([[3, 4], [4, 3]], dtype=torch.long),        # Graph 2
                torch.tensor([[5, 6, 7], [6, 7, 5]], dtype=torch.long)  # Graph 3
            ],
            'batch_info': torch.tensor([0, 0, 0, 1, 1, 2, 2, 2] + [0]*42),  # Graph assignments
            'targets': torch.randn(50, 1)
        }

        batch_result = model.process_graph_batch(graph_batch)

        assert batch_result is not None
        assert 'batch_predictions' in batch_result
        assert 'individual_graph_outputs' in batch_result


class TestModelInterfaceCompatibility:
    def test_should_distinguish_learning_paradigms(self):
        """Test that models correctly identify their learning paradigm"""
        transductive_config = TransductiveModelConfig(input_dim=4, output_dim=1, num_nodes=10)
        inductive_config = InductiveModelConfig(input_dim=4, output_dim=1)

        transductive_model = TransductiveGNNModel(transductive_config)
        inductive_model = InductiveGNNModel(inductive_config)

        assert transductive_model.get_learning_paradigm() == "transductive"
        assert inductive_model.get_learning_paradigm() == "inductive"

    def test_should_have_compatible_base_interface(self):
        """Test that both models share common base interface"""
        transductive_config = TransductiveModelConfig(input_dim=6, output_dim=1, num_nodes=15)
        inductive_config = InductiveModelConfig(input_dim=6, output_dim=1)

        transductive_model = TransductiveGNNModel(transductive_config)
        inductive_model = InductiveGNNModel(inductive_config)

        # Both should have base interface methods
        assert hasattr(transductive_model, 'get_model_summary')
        assert hasattr(inductive_model, 'get_model_summary')
        assert hasattr(transductive_model, 'save_model')
        assert hasattr(inductive_model, 'save_model')

    def test_should_validate_model_selection_criteria(self):
        """Test criteria for choosing between transductive and inductive models"""
        # Scenario 1: Fixed graph structure -> Transductive
        fixed_graph_scenario = {
            'has_fixed_nodes': True,
            'node_set_changes': False,
            'supports_semi_supervised': True
        }

        paradigm = self._determine_learning_paradigm(fixed_graph_scenario)
        assert paradigm == "transductive"

        # Scenario 2: Dynamic graph with new nodes -> Inductive
        dynamic_graph_scenario = {
            'has_fixed_nodes': False,
            'node_set_changes': True,
            'needs_generalization': True
        }

        paradigm = self._determine_learning_paradigm(dynamic_graph_scenario)
        assert paradigm == "inductive"

    def _determine_learning_paradigm(self, scenario: Dict[str, bool]) -> str:
        """Helper method to determine appropriate learning paradigm"""
        if scenario.get('has_fixed_nodes') and not scenario.get('node_set_changes'):
            return "transductive"
        elif scenario.get('node_set_changes') or scenario.get('needs_generalization'):
            return "inductive"
        else:
            return "hybrid"