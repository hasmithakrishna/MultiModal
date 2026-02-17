# Oxmaint Predictive Maintenance Agent

A production-ready, multi-modal AI system for predicting pump failures using sensor data, maintenance records, and visual inspection data.

## Features

- **Multi-Modal Analysis**: Combines sensor time-series, text documents, and images
- **Real-Time Prediction**: REST API with <500ms response time
- **Graceful Degradation**: Handles missing modalities without failure
- **Production Ready**: Docker containerized, health checks, monitoring
- **Scalable Architecture**: Modular design for easy extension

## Sample System Output

```json
{
  "asset_id": "pump_017",
  "failure_probability": 0.79,
  "estimated_time_to_breakdown_hours": 42.0,
  "predicted_fault_type": "bearing_failure",
  "fault_confidence": 0.73,
  "top_signals": ["vibration_spike", "maintenance_history_alert"],
  "inference_ms": 510,
  "model_version": "v1.0.0"
}
```

## Architecture

### Components

1. **Data Pipeline**: Ingestion and normalization of multi-modal data
2. **Preprocessing**: Windowing, scaling, feature engineering
3. **Models**:
   - Sensor: LSTM with attention mechanism
   - Text: Sentence transformers + keyword extraction
   - Image: CNN for defect detection (extensible)
4. **Orchestrator**: Multi-modal fusion and inference routing
5. **API**: FastAPI service with batch support

### Data Flow

```
Input → Validation → Preprocessing → Model Inference → Fusion → Output
                                    ↓
                            [Sensor Model]
                            [Text Model]
                            [Image Model]
```


### Data Preparation

```bash
# Configure Kaggle credentials
mkdir -p ~/.kaggle
# Add your kaggle.json to ~/.kaggle/

# Download will happen during setup
```

#### Option 2: Manual Download

1. **Sensor Data**: Download from [Kaggle](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)
   - Extract `sensor.csv` to `data/raw/sensor/`

2. **Image Data**: Download from [Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
   - Extract to `data/raw/images/casting_data/`

3. **Text Data**: Auto-generated during setup
   - Run `python src/data_pipeline/generate_mock_text.py` if needed