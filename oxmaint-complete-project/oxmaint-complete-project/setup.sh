#!/bin/bash
# Master setup script for Oxmaint Predictive Agent

set -e  # Exit on error

echo "========================================="
echo "Oxmaint Predictive Agent Setup"
echo "========================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Create virtual environment
echo ""
echo "Step 1: Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Step 2: Activate virtual environment
echo ""
echo "Step 2: Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Step 3: Upgrade pip
echo ""
echo "Step 3: Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"

# Step 4: Install dependencies
echo ""
echo "Step 4: Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Step 5: Create .env file
echo ""
echo "Step 5: Setting up configuration..."
if [ ! -f ".env" ]; then
    cp .env.template .env
    echo "✓ .env file created from template"
else
    echo "✓ .env file already exists"
fi

# Step 6: Generate mock text data
echo ""
echo "Step 6: Generating mock text data..."
python src/data_pipeline/generate_mock_text.py
echo "✓ Mock text data generated"

# Step 6: Generate mock text data
echo ""
echo "Step 6: Generating mock text data..."
python src/data_pipeline/generate_mock_text.py
echo "✓ Mock text data generated"

# Step 7: Setup Kaggle and download datasets
echo ""
echo "Step 7: Setting up Kaggle and downloading datasets..."
echo "Note: Place your kaggle.json file in the project directory"
echo ""

if [ -f "kaggle.json" ]; then
    echo "✓ Found kaggle.json file"
    python src/data_pipeline/setup_kaggle.py
else
    echo "⚠ kaggle.json not found in project directory"
    echo ""
    echo "Please download kaggle.json from: https://www.kaggle.com/settings"
    echo "Then place it in this directory and run:"
    echo "  python src/data_pipeline/setup_kaggle.py"
    echo ""
    echo "Or manually download datasets:"
    echo "  1. Sensor: https://www.kaggle.com/datasets/nphantawee/pump-sensor-data"
    echo "  2. Images: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. If you manually downloaded data, extract it to the correct directories"
echo "2. Run preprocessing: bash run_preprocessing.sh"
echo "3. Train models: bash run_training.sh"
echo "4. Start API server: bash run_api.sh"
echo ""
echo "Or run everything at once: bash run_all.sh"
echo ""
