# External Services Specification for GNN Update System

## Overview
This document outlines the external services that need to be built separately from the main model service repository to support the GNN model update requirements.

## Service Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│                     │    │                      │    │                     │
│   Market Event      │───▶│   Update Decision    │───▶│   Model Service     │
│   Monitor Service   │    │   Engine Service     │    │   (This Repo)       │
│                     │    │                      │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│                     │    │                      │    │                     │
│   Data Pipeline     │    │   Notification       │    │   Model Update      │
│   Coordinator       │    │   Service            │    │   API Endpoints     │
│                     │    │                      │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

## 1. Market Event Monitor Service

### Purpose
Continuously monitor market events and data changes that should trigger model updates.

### Repository: `stock-ai-event-monitor`

### Core Responsibilities
- **New Stock Detection**
  - Monitor stock exchanges for IPOs and new listings
  - Track when new stocks become available in data feeds
  - Identify delisting events

- **Key Stock Event Monitoring**
  - Earnings announcements and dates
  - Major corporate news and press releases
  - Rating changes from agencies (S&P, Moody's, etc.)
  - Insider trading notifications
  - Merger & acquisition announcements

- **Market Regime Detection**
  - Volatility spike detection
  - Sector rotation identification
  - Market crash or surge patterns
  - Economic indicator releases (GDP, unemployment, etc.)

### Technical Stack
- **Event Streaming**: Apache Kafka or AWS Kinesis
- **Data Sources**:
  - Financial APIs (Bloomberg, Reuters, Alpha Vantage)
  - News APIs (NewsAPI, Financial Times API)
  - SEC filing monitors
- **Storage**: PostgreSQL for event history
- **Language**: Python with asyncio for high-throughput event processing

### API Contract
```python
# Webhook to Update Decision Engine
POST /events/stock-event
{
  "event_type": "new_stock" | "earnings" | "rating_change" | "major_news",
  "symbols": ["AAPL", "GOOGL"],
  "severity": "low" | "medium" | "high" | "critical",
  "timestamp": "2024-01-01T10:00:00Z",
  "metadata": {
    "source": "bloomberg",
    "details": "..."
  }
}
```

## 2. Update Decision Engine Service

### Purpose
Analyze incoming events and decide whether to trigger inductive or transductive learning updates.

### Repository: `stock-ai-update-engine`

### Core Responsibilities
- **Update Type Decision Logic**
  ```python
  def decide_update_type(events, current_model_age, affected_stocks_count):
      if affected_stocks_count < 10:
          return "inductive"  # Short-term, partial update
      elif current_model_age > 30_days:
          return "transductive"  # Long-term, full retrain
      else:
          return "defer"  # Wait for more events or time
  ```

- **Priority Scoring**
  - Weight events by market impact
  - Consider computational resources
  - Factor in model performance degradation

- **Resource Management**
  - Check GPU/CPU availability
  - Estimate update duration and costs
  - Queue management for multiple updates

- **Update Scheduling**
  - Batch similar updates together
  - Avoid updates during market hours (if needed)
  - Handle emergency updates (market crashes)

### Decision Matrix
| Event Type | Affected Stocks | Model Age | Decision | Priority |
|------------|-----------------|-----------|----------|----------|
| New Stock | 1-5 | Any | Inductive | Medium |
| Earnings | 1-20 | <30 days | Inductive | Low |
| Sector Shift | 50+ | <30 days | Inductive (Cluster) | High |
| Market Crash | All | Any | Transductive | Critical |
| Normal Operations | All | >30 days | Transductive | Low |

### Technical Stack
- **Framework**: FastAPI for REST endpoints
- **Rule Engine**: Python-rules or Drools
- **Queue System**: Redis or RabbitMQ
- **Database**: PostgreSQL for decision history
- **Monitoring**: Prometheus + Grafana

### API Contracts
```python
# Receive events from Monitor Service
POST /events/analyze
{
  "events": [...],
  "context": {
    "model_last_updated": "2024-01-01T00:00:00Z",
    "current_performance": 0.85,
    "resource_availability": "high"
  }
}

# Response
{
  "decision": "inductive" | "transductive" | "defer",
  "priority": "low" | "medium" | "high" | "critical",
  "affected_symbols": ["AAPL", "GOOGL"],
  "estimated_duration": "15 minutes",
  "scheduled_time": "2024-01-01T11:00:00Z"
}
```

## 3. Data Pipeline Coordinator Service

### Purpose
Ensure fresh, validated data is available before triggering model updates.

### Repository: `stock-ai-data-coordinator`

### Core Responsibilities
- **Data Freshness Validation**
  - Check last update timestamps
  - Validate data completeness
  - Identify missing symbols or dates

- **Data Quality Checks**
  - Run validation pipelines
  - Check for anomalies or outliers
  - Verify data consistency

- **Update Readiness Assessment**
  - Confirm all required data is available
  - Check data processing pipeline status
  - Validate feature engineering completion

### Integration Points
- Works with existing ETL pipeline (Phase 1)
- Integrates with data validation (Phase 1)
- Coordinates with feature engineering (Phase 1)

## 4. Notification Service

### Purpose
Handle notifications and status updates throughout the update process.

### Repository: `stock-ai-notifications`

### Core Responsibilities
- **Status Broadcasting**
  - Update start/completion notifications
  - Progress tracking during long updates
  - Error and failure alerts

- **Integration Notifications**
  - Slack/Teams integration for team updates
  - Email alerts for critical issues
  - Dashboard updates for monitoring

- **Audit Trail**
  - Log all update decisions and outcomes
  - Track performance changes after updates
  - Maintain update history for analysis

## Model Service Integration Points

### Required API Endpoints (This Repo)
```python
# Trigger inductive update
POST /model/update/inductive
{
  "symbols": ["AAPL", "GOOGL"],
  "update_type": "new_stocks" | "key_stocks" | "cluster",
  "cluster_id": "tech_sector" (optional),
  "priority": "low" | "medium" | "high"
}

# Trigger transductive update
POST /model/update/transductive
{
  "full_retrain": true,
  "target_date": "2024-01-01T00:00:00Z",
  "priority": "low" | "medium" | "high"
}

# Get update status
GET /model/update/status/{update_id}

# Get model version info
GET /model/version/current

# Rollback model version
POST /model/rollback
{
  "target_version": "v1.2.3",
  "reason": "Performance degradation"
}
```

## Deployment Strategy

### Development Phase
1. **Phase 1**: Build Event Monitor Service
2. **Phase 2**: Build Update Decision Engine
3. **Phase 3**: Integrate with Model Service APIs
4. **Phase 4**: Add Data Coordinator and Notifications

### Production Deployment
- **Containerization**: Docker containers for each service
- **Orchestration**: Kubernetes for scaling and management
- **Service Mesh**: Istio for inter-service communication
- **Monitoring**: Centralized logging and metrics

## Estimated Timeline

| Service | Development Time | Integration Time |
|---------|------------------|------------------|
| Market Event Monitor | 3-4 weeks | 1 week |
| Update Decision Engine | 2-3 weeks | 1 week |
| Data Pipeline Coordinator | 2 weeks | 1 week |
| Notification Service | 1-2 weeks | 1 week |
| **Total** | **8-11 weeks** | **2-3 weeks** |

## Success Metrics

### Operational Metrics
- **Update Latency**: Time from event detection to model update completion
- **Decision Accuracy**: Correctness of inductive vs transductive decisions
- **Resource Utilization**: Efficient use of computational resources
- **Update Success Rate**: Percentage of successful updates

### Business Metrics
- **Model Performance**: Improved prediction accuracy after updates
- **System Availability**: Minimal downtime during updates
- **Cost Efficiency**: Reduced computational costs vs. performance gains

## Risk Mitigation

### Technical Risks
- **Event Processing Delays**: Implement event buffering and replay mechanisms
- **Service Dependencies**: Design for graceful degradation when services are unavailable
- **Data Quality Issues**: Comprehensive validation before triggering updates

### Business Risks
- **Over-updating**: Prevent excessive updates that could destabilize the model
- **Under-updating**: Ensure critical events trigger appropriate responses
- **Performance Regression**: Robust rollback mechanisms and A/B testing

---

**Next Steps**:
1. Review and approve this specification
2. Create GitHub repositories for external services
3. Begin development with Market Event Monitor Service
4. Implement Model Service API endpoints in parallel