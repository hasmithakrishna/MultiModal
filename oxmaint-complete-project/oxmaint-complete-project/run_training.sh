#!/bin/bash
# Run model training

set -e

echo "Running Model Training..."
echo ""

# Activate venv
source venv/bin/activate

# Train sensor model (primary model)
echo "Training sensor LSTM model..."
python src/modeling/train_sensor_model.py

echo ""
echo "✓ Model training complete!"
echo "Models saved to: models/"
