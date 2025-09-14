# Modular Model Architecture Specification

## Overview
Design a modular, replaceable GNN model architecture that allows easy swapping of different model types (GCN, GraphSAGE, GAT, etc.) without changing the core system infrastructure.

## Design Principles

### 1. **Model Abstraction Layer**
All GNN models must implement a common interface to ensure interchangeability.

### 2. **Plugin Architecture**
Models are loaded as plugins through a model registry system.

### 3. **Configuration-Driven**
Model selection and parameters are controlled through configuration files.

### 4. **Backward Compatibility**
New models must be compatible with existing data pipelines and APIs.

## Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Registry & Factory                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   GCN Plugin    │  │ GraphSAGE Plugin│  │   GAT Plugin    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                     Base Model Interface                        │
├─────────────────────────────────────────────────────────────────┤
│                   Model Configuration Manager                    │
├─────────────────────────────────────────────────────────────────┤
│                     Model Version Manager                       │
└─────────────────────────────────────────────────────────────────┘
```

## Required Interfaces

### Base Model Interface
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import torch
from enum import Enum

class LearningType(Enum):
    TRANSDUCTIVE = "transductive"
    INDUCTIVE = "inductive"
    HYBRID = "hybrid"

class BaseGNNModel(ABC):
    """Base interface for all GNN models"""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        self.learning_type = LearningType(config.get('learning_type', 'transductive'))
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings for analysis"""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> bool:
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> bool:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and configuration"""
        pass

    @property
    def supports_inductive_learning(self) -> bool:
        """Check if model supports inductive learning"""
        return self.learning_type in [LearningType.INDUCTIVE, LearningType.HYBRID]

    @property
    def supports_transductive_learning(self) -> bool:
        """Check if model supports transductive learning"""
        return self.learning_type in [LearningType.TRANSDUCTIVE, LearningType.HYBRID]
```

### Transductive Model Interface
```python
class TransductiveGNNModel(BaseGNNModel):
    """Interface for transductive learning models (like GCN)

    Transductive models:
    - Learn from the entire graph during training
    - Cannot generalize to completely unseen nodes
    - Require full graph retraining for new nodes
    - Optimal for scenarios where all nodes are known upfront
    """

    @abstractmethod
    def full_retrain(self, x: torch.Tensor, edge_index: torch.Tensor,
                     edge_weight: Optional[torch.Tensor] = None,
                     y: Optional[torch.Tensor] = None) -> bool:
        """Complete model retraining with full graph"""
        pass

    @abstractmethod
    def update_graph_structure(self, new_edge_index: torch.Tensor,
                              new_edge_weight: Optional[torch.Tensor] = None) -> bool:
        """Update model with new graph structure (requires retraining)"""
        pass

    @abstractmethod
    def get_all_node_embeddings(self) -> torch.Tensor:
        """Get embeddings for all nodes in the training graph"""
        pass

    def add_new_nodes(self, new_node_features: torch.Tensor,
                      updated_edge_index: torch.Tensor) -> bool:
        """Add new nodes - requires full retraining for transductive models"""
        # Transductive models need full retraining when new nodes are added
        return self.full_retrain(new_node_features, updated_edge_index)

    def partial_update(self, node_ids: List[int], new_features: torch.Tensor) -> bool:
        """Partial updates not supported in pure transductive models"""
        raise NotImplementedError("Transductive models require full retraining for updates")
```

### Inductive Model Interface
```python
class InductiveGNNModel(BaseGNNModel):
    """Interface for inductive learning models (like GraphSAGE)

    Inductive models:
    - Learn generalizable patterns from local neighborhoods
    - Can generate embeddings for completely unseen nodes
    - Support efficient partial updates
    - Optimal for dynamic graphs with new nodes
    """

    @abstractmethod
    def inductive_forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                         edge_weight: Optional[torch.Tensor] = None,
                         target_nodes: Optional[List[int]] = None) -> torch.Tensor:
        """Forward pass for specific target nodes (inductive)"""
        pass

    @abstractmethod
    def partial_update(self, node_ids: List[int], new_features: torch.Tensor) -> bool:
        """Update embeddings for specific nodes without full retraining"""
        pass

    @abstractmethod
    def add_new_nodes(self, new_node_features: torch.Tensor,
                      updated_edge_index: torch.Tensor,
                      new_node_ids: List[int]) -> torch.Tensor:
        """Add new nodes and generate embeddings without retraining"""
        pass

    @abstractmethod
    def cluster_update(self, cluster_nodes: List[int],
                      cluster_features: torch.Tensor,
                      cluster_edges: torch.Tensor) -> bool:
        """Update specific cluster/sector of nodes"""
        pass

    @abstractmethod
    def get_node_embeddings(self, target_node_ids: List[int]) -> torch.Tensor:
        """Get embeddings for specific nodes"""
        pass

    def supports_online_learning(self) -> bool:
        """Check if model supports online/streaming updates"""
        return True
```

### Hybrid Model Interface
```python
class HybridGNNModel(TransductiveGNNModel, InductiveGNNModel):
    """Interface for models that support both learning paradigms

    Hybrid models:
    - Can operate in both transductive and inductive modes
    - Switch between modes based on scenario requirements
    - Provide optimal performance for different update patterns
    """

    @abstractmethod
    def set_learning_mode(self, mode: LearningType) -> bool:
        """Switch between transductive and inductive learning modes"""
        pass

    @abstractmethod
    def get_current_mode(self) -> LearningType:
        """Get current learning mode"""
        pass

    @abstractmethod
    def optimize_for_scenario(self, scenario: str) -> LearningType:
        """Automatically choose optimal learning mode for scenario

        Scenarios:
        - 'new_stocks': Inductive mode for adding new stocks
        - 'full_retrain': Transductive mode for comprehensive updates
        - 'cluster_update': Inductive mode for sectoral updates
        - 'market_regime_change': Transductive mode for structural changes
        """
        pass
```

### Model Registry System
```python
class ModelRegistry:
    """Registry for available GNN models"""

    def register_model(self, name: str, model_class: type,
                      config_schema: Dict[str, Any]) -> None:
        pass

    def get_model(self, name: str, config: Dict[str, Any]) -> BaseGNNModel:
        pass

    def list_available_models(self) -> List[str]:
        pass

    def validate_config(self, name: str, config: Dict[str, Any]) -> bool:
        pass
```

### Model Factory
```python
class ModelFactory:
    """Factory for creating model instances"""

    def create_model(self, model_name: str, config: Dict[str, Any]) -> BaseGNNModel:
        pass

    def create_from_config_file(self, config_path: str) -> BaseGNNModel:
        pass

    def clone_model(self, source_model: BaseGNNModel) -> BaseGNNModel:
        pass
```

## Updated Plan Requirements

### Phase 3 Additions (Model Modularity)
- [ ] **Design and implement base GNN model interface**
- [ ] **Create model registry and factory system**
- [ ] **Implement model plugin architecture**
- [ ] **Add model configuration management system**
- [ ] Implement basic GCN model with PyTorch Geometric
- [ ] Add GraphSAGE implementation for scalability (inductive learning)
- [ ] **Create model adapter layer for legacy compatibility**
- [ ] Implement inductive learning framework for new nodes
- [ ] Add partial model update mechanisms for targeted retraining
- [ ] Create cluster-based update system for sectoral changes
- [ ] Create temporal encoding mechanisms
- [ ] Implement attention-based architectures
- [ ] **Add hot-swapping capabilities for live model replacement**
- [ ] Add model checkpointing and versioning
- [ ] Create model evaluation framework

### Phase 4 Additions (Model Management)
- [ ] **Implement A/B testing framework for model comparison**
- [ ] **Add model performance monitoring and auto-switching**
- [ ] **Create model rollback mechanisms**
- [ ] Implement Granger causality testing suite
- [ ] Add transfer entropy calculations
- [ ] Create multi-horizon prediction framework
- [ ] Add inductive vs transductive learning router
- [ ] Implement incremental training pipeline for short-term updates
- [ ] Create transductive learning pipeline for long-term updates
- [ ] Implement ensemble methods
- [ ] Add uncertainty quantification
- [ ] Build backtesting framework

### Phase 5 Additions (Production Model Management)
- [ ] **Add model deployment pipeline with zero-downtime switching**
- [ ] **Implement model canary deployments**
- [ ] **Add automated model selection based on performance metrics**
- [ ] Hyperparameter optimization with Optuna
- [ ] Model compression and quantization
- [ ] Implement real-time inference pipeline
- [ ] Add model update management system
- [ ] Implement A/B testing framework for model versions
- [ ] Add model rollback and version control mechanisms
- [ ] Create update timing and scheduling system
- [ ] Add monitoring and alerting systems
- [ ] Create REST API with FastAPI (including update endpoints)
- [ ] Set up Docker containerization

## Configuration Schema

### Transductive Model Configuration
```yaml
# config/models/transductive/gcn_config.yml
model:
  name: "gcn_transductive"
  version: "1.0.0"
  learning_type: "transductive"
  category: "transductive"

architecture:
  input_dim: 128
  hidden_dims: [256, 128, 64]
  output_dim: 32
  num_layers: 3
  dropout: 0.2
  activation: "relu"
  normalization: "batch_norm"

training:
  learning_rate: 0.001
  batch_size: 512        # Larger batches for transductive
  epochs: 200           # More epochs for full graph learning
  optimizer: "adam"
  scheduler: "cosine"
  full_graph_training: true

capabilities:
  inductive_learning: false
  partial_updates: false
  requires_full_retrain: true
  supports_all_nodes: true
  temporal_encoding: true
  attention_mechanism: false

update_strategy:
  type: "full_retrain"
  frequency: "monthly"    # Long-term updates
  triggers: ["market_regime_change", "quarterly_retrain"]

use_cases:
  - "comprehensive_market_analysis"
  - "long_term_relationship_modeling"
  - "stable_market_periods"

metadata:
  description: "Transductive GCN for comprehensive graph learning"
  author: "stock.ai team"
  created_date: "2024-01-01"
  optimal_scenarios: ["full_market_analysis", "batch_processing"]
```

### Inductive Model Configuration
```yaml
# config/models/inductive/graphsage_config.yml
model:
  name: "graphsage_inductive"
  version: "1.0.0"
  learning_type: "inductive"
  category: "inductive"

architecture:
  input_dim: 128
  hidden_dims: [256, 128, 64]
  output_dim: 32
  num_layers: 2         # Fewer layers for efficiency
  dropout: 0.3
  activation: "relu"
  aggregator: "mean"    # mean, max, lstm, pool
  sampling_sizes: [15, 10]  # Neighborhood sampling

training:
  learning_rate: 0.005  # Higher learning rate
  batch_size: 256       # Moderate batch size
  epochs: 100           # Fewer epochs needed
  optimizer: "adam"
  scheduler: "step"
  neighborhood_sampling: true
  num_neighbors: [15, 10]

capabilities:
  inductive_learning: true
  partial_updates: true
  requires_full_retrain: false
  supports_new_nodes: true
  online_learning: true
  cluster_updates: true

update_strategy:
  type: "partial_update"
  frequency: "daily"     # Short-term updates
  triggers: ["new_stocks", "sector_updates", "earnings_events"]

sampling:
  strategy: "uniform"    # uniform, importance, fastgcn
  num_layers: 2
  sample_sizes: [15, 10]

use_cases:
  - "new_stock_integration"
  - "real_time_updates"
  - "sector_specific_analysis"
  - "streaming_data_processing"

metadata:
  description: "Inductive GraphSAGE for dynamic graph learning"
  author: "stock.ai team"
  created_date: "2024-01-01"
  optimal_scenarios: ["new_node_addition", "partial_updates", "real_time_inference"]
```

### Hybrid Model Configuration
```yaml
# config/models/hybrid/adaptive_gnn_config.yml
model:
  name: "adaptive_gnn_hybrid"
  version: "1.0.0"
  learning_type: "hybrid"
  category: "hybrid"

architecture:
  input_dim: 128
  hidden_dims: [256, 128, 64]
  output_dim: 32
  shared_layers: 2      # Layers shared between modes
  transductive_layers: 1 # Additional layers for transductive mode
  inductive_layers: 1    # Additional layers for inductive mode
  mode_switch_layer: true

modes:
  transductive:
    enabled: true
    default_config: "gcn_transductive"

  inductive:
    enabled: true
    default_config: "graphsage_inductive"

  auto_switch:
    enabled: true
    decision_threshold: 0.1  # Performance difference threshold
    evaluation_window: "7days"

training:
  # Will inherit from active mode configuration
  adaptive_learning_rate: true
  mode_specific_optimizers: true

capabilities:
  inductive_learning: true
  partial_updates: true
  full_retrain: true
  mode_switching: true
  auto_optimization: true

update_strategy:
  type: "adaptive"
  decision_rules:
    new_stocks_count:
      "<10": "inductive"
      ">=10": "evaluate"
      ">50": "transductive"

    time_since_retrain:
      "<7days": "inductive"
      "7-30days": "evaluate"
      ">30days": "transductive"

    market_volatility:
      "high": "inductive"  # Quick adaptations
      "low": "transductive"  # Stable learning

metadata:
  description: "Adaptive hybrid model switching between learning paradigms"
  author: "stock.ai team"
  created_date: "2024-01-01"
  optimal_scenarios: ["variable_market_conditions", "mixed_update_patterns"]
```

## Model Plugin Structure

```
src/models/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── base_model.py          # BaseGNNModel interface
│   ├── transductive_model.py  # TransductiveGNNModel interface
│   ├── inductive_model.py     # InductiveGNNModel interface
│   ├── hybrid_model.py        # HybridGNNModel interface
│   ├── model_registry.py      # Model registry
│   └── model_factory.py       # Model factory
├── transductive/              # Transductive learning models
│   ├── __init__.py
│   ├── gcn/
│   │   ├── __init__.py
│   │   ├── gcn_model.py       # GCN transductive implementation
│   │   ├── gcn_config.py      # GCN configuration
│   │   └── gcn_plugin.py      # Plugin registration
│   ├── gat/
│   │   ├── __init__.py
│   │   ├── gat_model.py       # GAT transductive implementation
│   │   └── gat_config.py
│   └── gin/                   # Graph Isomorphism Networks
├── inductive/                 # Inductive learning models
│   ├── __init__.py
│   ├── graphsage/
│   │   ├── __init__.py
│   │   ├── graphsage_model.py # GraphSAGE inductive implementation
│   │   ├── graphsage_config.py
│   │   └── graphsage_plugin.py
│   ├── fastgcn/
│   │   ├── __init__.py
│   │   └── fastgcn_model.py   # FastGCN for large graphs
│   └── graphsaint/            # GraphSAINT sampling-based
├── hybrid/                    # Hybrid models supporting both paradigms
│   ├── __init__.py
│   ├── adaptive_gnn/
│   │   ├── __init__.py
│   │   ├── adaptive_model.py  # Adaptive GNN (switches modes)
│   │   └── adaptive_config.py
│   └── universal_gnn/         # Universal architecture
└── custom/                    # Custom user models
    ├── transductive/
    ├── inductive/
    └── hybrid/
```

## Model Lifecycle Management

### 1. Model Registration
```python
# Register a new model
registry = ModelRegistry()
registry.register_model(
    name="custom_gnn",
    model_class=CustomGNNModel,
    config_schema=custom_config_schema
)
```

### 2. Model Creation
```python
# Create model instance
factory = ModelFactory()
model = factory.create_model("gcn", config_dict)
```

### 3. Model Switching
```python
# Hot-swap models in production
model_manager = ModelManager()
model_manager.switch_model(
    from_model="gcn_v1",
    to_model="graphsage_v2",
    strategy="gradual"  # or "immediate"
)
```

### 4. Model Comparison
```python
# A/B testing
ab_tester = ModelABTester()
results = ab_tester.compare_models(
    model_a="gcn",
    model_b="graphsage",
    test_duration="7days",
    traffic_split=0.5
)
```

## Benefits of Modular Design

### 1. **Easy Model Replacement**
- Swap models without code changes
- Configuration-driven model selection
- Zero-downtime model updates

### 2. **Extensibility**
- Add new model types easily
- Custom model implementations
- Third-party model integration

### 3. **Testing & Validation**
- A/B testing between models
- Performance benchmarking
- Gradual rollouts

### 4. **Maintenance**
- Independent model updates
- Version control per model
- Rollback capabilities

### 5. **Development Efficiency**
- Parallel model development
- Standardized interfaces
- Reusable components

## Implementation Priority

### High Priority (Phase 3)
1. Base model interface design
2. Model registry and factory
3. GCN and GraphSAGE plugins
4. Configuration management

### Medium Priority (Phase 4)
1. A/B testing framework
2. Model performance monitoring
3. Hot-swapping capabilities

### Low Priority (Phase 5)
1. Advanced deployment strategies
2. Automated model selection
3. Canary deployments

This modular architecture ensures that the GNN system can evolve and adapt to new model architectures while maintaining stability and backward compatibility.