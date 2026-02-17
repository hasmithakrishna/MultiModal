#!/bin/bash
# Run all preprocessing steps

set -e

echo "Running Preprocessing Pipeline..."
echo ""

# Activate venv
source venv/bin/activate

# Step 1: Sensor preprocessing
echo "1/3: Preprocessing sensor data..."
python src/preprocessing/sensor_preprocessing.py

# Step 2: Text preprocessing
echo ""
echo "2/3: Preprocessing text data..."
python src/preprocessing/text_preprocessing.py

# Step 3: Image preprocessing
echo ""
echo "3/3: Preprocessing image data..."
python src/preprocessing/image_preprocessing.py

echo ""
echo "✓ All preprocessing complete!"
