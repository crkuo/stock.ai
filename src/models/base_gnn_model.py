import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration class for GNN models"""
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = None
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.01
    batch_size: int = 32
    weight_decay: float = 1e-5

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


@dataclass
class ModelMetadata:
    """Metadata class for model information"""
    model_name: str
    version: str
    created_at: datetime
    description: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class BaseGNNModel(nn.Module):
    """Base class for all GNN models"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.metadata: Optional[ModelMetadata] = None
        self._is_trained = False

        # Initialize model architecture
        self._init_model()

    def _init_model(self):
        """Initialize the model architecture - to be implemented by subclasses"""
        # Simple linear layer for base implementation
        self.layers = nn.ModuleList()

        # Input layer
        prev_dim = self.config.input_dim
        for hidden_dim in self.config.hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.config.dropout))
            prev_dim = hidden_dim

        # Output layer
        self.layers.append(nn.Linear(prev_dim, self.config.output_dim))

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    def set_metadata(self, metadata: ModelMetadata):
        """Set model metadata"""
        self.metadata = metadata

    def get_metadata(self) -> Optional[ModelMetadata]:
        """Get model metadata"""
        return self.metadata

    def validate_input(self, node_features: torch.Tensor,
                      edge_index: torch.Tensor,
                      edge_weights: Optional[torch.Tensor] = None) -> bool:
        """Validate input tensors"""
        # Check node features dimension
        if node_features.shape[1] != self.config.input_dim:
            return False

        # Check edge_index format
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            return False

        # Check edge weights if provided
        if edge_weights is not None:
            if edge_weights.shape[0] != edge_index.shape[1]:
                return False

        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - simple implementation for base class"""
        for layer in self.layers:
            x = layer(x)
        return x

    def train_model(self, train_data: Dict[str, torch.Tensor],
                   validation_data: Dict[str, torch.Tensor],
                   epochs: int = 100) -> Dict[str, Any]:
        """Train the model"""
        optimizer = torch.optim.Adam(self.parameters(),
                                   lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        for _ in range(epochs):
            # Training
            self.train()
            optimizer.zero_grad()

            outputs = self(train_data['node_features'])
            loss = criterion(outputs, train_data['targets'])
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Validation
            self.eval()
            with torch.no_grad():
                val_outputs = self(validation_data['node_features'])
                val_loss = criterion(val_outputs, validation_data['targets'])
                val_losses.append(val_loss.item())

        self._is_trained = True

        return {
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'epochs_completed': epochs,
            'train_loss_history': train_losses,
            'val_loss_history': val_losses
        }

    def predict(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            predictions = self(data['node_features'])
        return predictions

    def evaluate(self, test_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(test_data)
        targets = test_data['targets']

        # Calculate metrics
        mse = nn.functional.mse_loss(predictions, targets).item()
        mae = nn.functional.l1_loss(predictions, targets).item()

        # R² score
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot).item()

        return {
            'mse': mse,
            'mae': mae,
            'r2_score': r2_score
        }

    def save_model(self, save_path: str) -> bool:
        """Save model to file"""
        try:
            model_data = {
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'metadata': self.metadata,
                'is_trained': self._is_trained
            }
            torch.save(model_data, save_path)
            return True
        except Exception:
            return False

    @classmethod
    def load_model(cls, load_path: str) -> Optional['BaseGNNModel']:
        """Load model from file"""
        try:
            model_data = torch.load(load_path)
            model = cls(config=model_data['config'])
            model.load_state_dict(model_data['model_state_dict'])
            model.metadata = model_data.get('metadata')
            model._is_trained = model_data.get('is_trained', False)
            return model
        except Exception:
            return None

    def update_config(self, new_config: ModelConfig) -> bool:
        """Update model configuration"""
        try:
            # Only allow certain config updates without reinitializing
            self.config.learning_rate = new_config.learning_rate
            self.config.dropout = new_config.dropout
            self.config.batch_size = new_config.batch_size
            self.config.weight_decay = new_config.weight_decay
            return True
        except Exception:
            return False

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        layer_info = []
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):
                layer_info.append({
                    'layer_idx': i,
                    'layer_type': layer.__class__.__name__,
                    'input_size': layer.weight.shape[1] if hasattr(layer.weight, 'shape') else None,
                    'output_size': layer.weight.shape[0] if hasattr(layer.weight, 'shape') else None
                })

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layer_info': layer_info,
            'config': self.config,
            'is_trained': self._is_trained
        }

    def is_trained(self) -> bool:
        """Check if model has been trained"""
        return self._is_trained

    def _set_trained_state(self, state: bool):
        """Set trained state (for testing purposes)"""
        self._is_trained = state

    def reset_model(self):
        """Reset model to untrained state"""
        self._init_model()
        self._is_trained = False