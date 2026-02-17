#!/bin/bash
# Complete end-to-end pipeline runner

set -e

echo "========================================="
echo "Oxmaint Complete Pipeline Execution"
echo "========================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Step 1: Data preprocessing
echo ""
echo "Step 1: Running preprocessing pipeline..."
bash run_preprocessing.sh

# Step 2: Model training
echo ""
echo "Step 2: Training models..."
bash run_training.sh

# Step 3: Run simple inference test
echo ""
echo "Step 3: Testing inference..."
python src/orchestrator/inference_orchestrator.py

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo ""
echo "To start the API server, run: bash run_api.sh"
echo ""
