import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .base_gnn_model import BaseGNNModel, ModelConfig


@dataclass
class TransductiveModelConfig(ModelConfig):
    """Configuration for transductive GNN models"""
    num_nodes: int = 100
    requires_full_graph: bool = True
    supports_node_addition: bool = False
    semi_supervised: bool = False
    fixed_graph_structure: bool = True


class TransductiveGNNModel(BaseGNNModel):
    """Transductive GNN model for fixed graph structures"""

    def __init__(self, config: TransductiveModelConfig):
        super().__init__(config)
        self.transductive_config = config

    def supports_node_addition(self) -> bool:
        """Check if model supports adding new nodes"""
        return self.transductive_config.supports_node_addition

    def get_learning_paradigm(self) -> str:
        """Get the learning paradigm of this model"""
        return "transductive"

    def validate_graph_structure(self, data: Dict[str, torch.Tensor]) -> bool:
        """Validate that graph structure matches expected format"""
        node_features = data.get('node_features')

        if node_features is None:
            return False

        # Check if number of nodes matches expected
        if node_features.shape[0] != self.transductive_config.num_nodes:
            return False

        # Check feature dimension
        if node_features.shape[1] != self.config.input_dim:
            return False

        return True

    def train_transductive(self, train_data: Dict[str, torch.Tensor],
                          validation_data: Dict[str, torch.Tensor],
                          epochs: int = 100) -> Dict[str, Any]:
        """Train model using transductive learning"""
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

            # Get training nodes only
            train_mask = train_data['train_mask']
            node_features = train_data['node_features']
            targets = train_data['targets']

            # Forward pass on all nodes, but only compute loss on training nodes
            outputs = self(node_features)
            train_outputs = outputs[train_mask]
            train_targets = targets[train_mask]

            loss = criterion(train_outputs, train_targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Validation phase
            self.eval()
            with torch.no_grad():
                val_mask = validation_data['val_mask']
                val_outputs = outputs[val_mask]
                val_targets = targets[val_mask]
                val_loss = criterion(val_outputs, val_targets)
                val_losses.append(val_loss.item())

        self._set_trained_state(True)

        return {
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'epochs_completed': epochs,
            'train_loss_history': train_losses,
            'val_loss_history': val_losses
        }

    def train_semi_supervised(self, data: Dict[str, torch.Tensor],
                            epochs: int = 100) -> Dict[str, Any]:
        """Train model using semi-supervised learning"""
        if not self.transductive_config.semi_supervised:
            raise ValueError("Model not configured for semi-supervised learning")

        optimizer = torch.optim.Adam(self.parameters(),
                                   lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        labeled_losses = []

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            # Get labeled nodes
            labeled_mask = data['labeled_mask']
            node_features = data['node_features']
            targets = data['targets']

            # Forward pass on all nodes
            outputs = self(node_features)

            # Compute loss only on labeled nodes
            labeled_outputs = outputs[labeled_mask]
            labeled_targets = targets[labeled_mask]

            loss = criterion(labeled_outputs, labeled_targets)
            loss.backward()
            optimizer.step()

            labeled_losses.append(loss.item())

        self._set_trained_state(True)

        # Get predictions for unlabeled nodes
        with torch.no_grad():
            self.eval()
            all_outputs = self(node_features)
            unlabeled_predictions = all_outputs[~labeled_mask]

        return {
            'labeled_loss': labeled_losses[-1],
            'epochs_completed': epochs,
            'labeled_loss_history': labeled_losses,
            'unlabeled_predictions': unlabeled_predictions
        }