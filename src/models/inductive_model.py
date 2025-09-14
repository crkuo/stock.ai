import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .base_gnn_model import BaseGNNModel, ModelConfig


@dataclass
class InductiveModelConfig(ModelConfig):
    """Configuration for inductive GNN models"""
    supports_new_nodes: bool = True
    neighborhood_sampling: bool = False
    max_neighbors: int = 10
    sampling_layers: List[int] = None
    supports_incremental_training: bool = True
    batch_multiple_graphs: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.sampling_layers is None:
            self.sampling_layers = [10, 5]


class InductiveGNNModel(BaseGNNModel):
    """Inductive GNN model for dynamic graphs with new nodes"""

    def __init__(self, config: InductiveModelConfig):
        super().__init__(config)
        self.inductive_config = config

    def supports_new_nodes(self) -> bool:
        """Check if model supports new unseen nodes"""
        return self.inductive_config.supports_new_nodes

    def get_learning_paradigm(self) -> str:
        """Get the learning paradigm of this model"""
        return "inductive"

    def train_inductive(self, train_data: Dict[str, torch.Tensor],
                       validation_data: Dict[str, torch.Tensor],
                       epochs: int = 100) -> Dict[str, Any]:
        """Train model using inductive learning"""
        optimizer = torch.optim.Adam(self.parameters(),
                                   lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training phase
            self.train()
            optimizer.zero_grad()

            # Process training data (can be variable size)
            train_outputs = self(train_data['node_features'])
            loss = criterion(train_outputs, train_data['targets'])
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Validation phase (different graph size allowed)
            self.eval()
            with torch.no_grad():
                val_outputs = self(validation_data['node_features'])
                val_loss = criterion(val_outputs, validation_data['targets'])
                val_losses.append(val_loss.item())

        self._set_trained_state(True)

        return {
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'epochs_completed': epochs,
            'train_loss_history': train_losses,
            'val_loss_history': val_losses
        }

    def sample_neighborhood(self, data: Dict[str, torch.Tensor],
                           target_nodes: torch.Tensor) -> Dict[str, Any]:
        """Sample neighborhood for target nodes"""
        if not self.inductive_config.neighborhood_sampling:
            # Return full data if sampling is disabled
            return {
                'subgraph_nodes': torch.arange(data['node_features'].shape[0]),
                'subgraph_edges': data['edge_index'],
                'sampled_features': data['node_features']
            }

        # Simple sampling implementation
        # In practice, this would use GraphSAINT or FastGCN sampling
        max_neighbors = self.inductive_config.max_neighbors
        edge_index = data['edge_index']

        # Find neighbors for target nodes
        subgraph_nodes = set(target_nodes.tolist())

        for target_node in target_nodes:
            # Find neighbors of target node
            neighbors = edge_index[1][edge_index[0] == target_node]
            # Limit to max_neighbors
            if len(neighbors) > max_neighbors:
                sampled_neighbors = neighbors[:max_neighbors]
            else:
                sampled_neighbors = neighbors

            subgraph_nodes.update(sampled_neighbors.tolist())

        subgraph_nodes = torch.tensor(list(subgraph_nodes))

        # Extract subgraph edges
        mask = torch.isin(edge_index[0], subgraph_nodes) & torch.isin(edge_index[1], subgraph_nodes)
        subgraph_edges = edge_index[:, mask]

        return {
            'subgraph_nodes': subgraph_nodes,
            'subgraph_edges': subgraph_edges,
            'sampled_features': data['node_features'][subgraph_nodes]
        }

    def predict_new_nodes(self, new_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict on completely new unseen nodes"""
        if not self.supports_new_nodes():
            raise ValueError("Model does not support predictions on new nodes")

        if not self.is_trained():
            raise ValueError("Model must be trained before making predictions")

        self.eval()
        with torch.no_grad():
            predictions = self(new_data['node_features'])
        return predictions

    def train_incremental(self, new_data: Dict[str, torch.Tensor],
                         learning_rate_decay: float = 1.0,
                         epochs: int = 10) -> Dict[str, Any]:
        """Train incrementally on new data"""
        if not self.inductive_config.supports_incremental_training:
            raise ValueError("Model not configured for incremental training")

        # Reduce learning rate for incremental training
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate * learning_rate_decay
        )
        criterion = nn.MSELoss()

        incremental_losses = []

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            outputs = self(new_data['node_features'])
            loss = criterion(outputs, new_data['targets'])
            loss.backward()
            optimizer.step()

            incremental_losses.append(loss.item())

        return {
            'incremental_loss': incremental_losses[-1],
            'epochs_completed': epochs,
            'loss_history': incremental_losses,
            'model_updated': True
        }

    def process_graph_batch(self, graph_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of multiple graphs"""
        if not self.inductive_config.batch_multiple_graphs:
            raise ValueError("Model not configured for graph batching")

        node_features = graph_batch['node_features']
        targets = graph_batch['targets']

        # Simple batched processing
        self.eval()
        with torch.no_grad():
            batch_predictions = self(node_features)

        # Split predictions by graph (simplified implementation)
        edge_indices = graph_batch['edge_indices']
        individual_outputs = []

        start_idx = 0
        for i, edge_index in enumerate(edge_indices):
            # Estimate graph size from edge indices
            graph_nodes = torch.unique(edge_index).max().item() + 1
            end_idx = start_idx + graph_nodes
            individual_outputs.append(batch_predictions[start_idx:end_idx])
            start_idx = end_idx

        return {
            'batch_predictions': batch_predictions,
            'individual_graph_outputs': individual_outputs
        }