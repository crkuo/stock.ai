# GNN Model Update Requirements Analysis

## Overview
Analysis of current project planning against the requirements for periodic GNN model updates with distinction between short-term (Inductive Learning) and long-term (Transductive Learning) approaches.

## Current Plan Assessment

### ✅ Already Covered in Current Plan
1. **Dynamic Graph Updates** (Phase 2) - ✅ **IMPLEMENTED**
   - Real-time graph structure updates
   - Rolling data windows
   - Structural change detection
   - New stock addition handling

2. **GNN Model Development** (Phase 3) - Planned
   - GraphSAGE implementation (ideal for inductive learning)
   - Model checkpointing and versioning
   - Temporal encoding mechanisms

3. **Model Infrastructure** (Phase 5) - Planned
   - Real-time inference pipeline
   - Model compression and quantization

### ❌ Missing Requirements for Inductive/Transductive Learning

#### Short-term Updates (Inductive Learning) - MISSING
1. **Inductive Learning Framework**
   - Ability to generate embeddings for new nodes without full retraining
   - Partial model updates for specific stock subsets
   - Cluster-based update mechanisms

2. **Event-Driven Update Triggers**
   - New stock addition detection
   - Key stock update identification
   - Sector/cluster change detection

3. **Incremental Training Pipeline**
   - Partial model retraining
   - Embedding update for specific nodes
   - Model state management

#### Long-term Updates (Transductive Learning) - MISSING
1. **Full Model Retraining Pipeline**
   - Complete graph reconstruction
   - Full transductive learning workflow
   - Historical data integration

2. **Model Versioning and Rollback**
   - Major version management
   - A/B testing framework
   - Rollback mechanisms

## Recommended Service Architecture

### This Model Service Repo Should Include:

#### Phase 3 Additions (GNN Model Development)
- [x] Implement basic GCN model with PyTorch Geometric
- [x] Add GraphSAGE implementation for scalability (CRITICAL for inductive learning)
- [ ] **NEW: Implement inductive learning framework**
- [ ] **NEW: Add partial model update mechanisms**
- [ ] **NEW: Create cluster-based update system**
- [ ] Create temporal encoding mechanisms
- [ ] Implement attention-based architectures
- [ ] Add model checkpointing and versioning
- [ ] Create model evaluation framework

#### Phase 4 Additions (Enhanced Prediction System)
- [ ] Implement Granger causality testing suite
- [ ] Add transfer entropy calculations
- [ ] Create multi-horizon prediction framework
- [ ] **NEW: Add inductive vs transductive learning router**
- [ ] **NEW: Implement incremental training pipeline**
- [ ] Implement ensemble methods
- [ ] Add uncertainty quantification
- [ ] Build backtesting framework

#### Phase 5 Additions (Production with Update Management)
- [ ] Hyperparameter optimization with Optuna
- [ ] Model compression and quantization
- [ ] Implement real-time inference pipeline
- [ ] **NEW: Add model update management system**
- [ ] **NEW: Implement A/B testing framework**
- [ ] **NEW: Add model rollback mechanisms**
- [ ] Add monitoring and alerting systems
- [ ] Create REST API with FastAPI
- [ ] **NEW: Add update timing and scheduling system**

### External Event Trigger Service Should Include:

#### Event Detection and Triggering
1. **Market Event Monitor**
   - New stock IPO/listing detection
   - Earnings announcement tracking
   - Major news event detection
   - Rating changes monitoring

2. **Update Decision Engine**
   - Inductive vs Transductive decision logic
   - Update priority scoring
   - Resource availability checking
   - Update scheduling optimization

3. **Trigger Orchestration**
   - API calls to model service for updates
   - Update queue management
   - Retry logic and error handling
   - Update status tracking

4. **Data Pipeline Coordination**
   - Fresh data availability checking
   - Data quality validation
   - Update readiness assessment

## Implementation Priority

### Immediate Additions to Current Plan (High Priority)
1. **Inductive Learning Framework** - Add to Phase 3
2. **Partial Update Mechanisms** - Add to Phase 4
3. **Update Management System** - Add to Phase 5

### External Service Development (Medium Priority)
1. **Event Trigger Service** - Separate repository
2. **Market Event Monitor** - Can be integrated with data pipeline
3. **Update Orchestration** - Separate microservice

## Technical Requirements Summary

### Model Service Core Capabilities Needed:
- ✅ Dynamic graph updates (already implemented)
- ❌ Inductive learning for new nodes
- ❌ Partial model retraining
- ❌ Cluster-based updates
- ❌ Model version management
- ❌ Update timing control

### External Service Integration Points:
- REST API endpoints for triggering updates
- Event webhook receivers
- Update status reporting
- Model performance monitoring

## Recommended Next Steps
1. Update plan.md with missing inductive/transductive learning requirements
2. Create external service specification document
3. Implement GraphSAGE with inductive capabilities first
4. Design update management API contracts
5. Plan external trigger service architecture