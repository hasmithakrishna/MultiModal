#!/bin/bash
# SIMPLE SETUP - Just run this after placing kaggle.json

set -e  # Exit on error

echo "========================================="
echo "OXMAINT SIMPLE SETUP"
echo "========================================="
echo ""

# Check for kaggle.json
if [ ! -f "kaggle.json" ]; then
    echo "ERROR: kaggle.json not found!"
    echo ""
    echo "Please download your Kaggle API token:"
    echo "  1. Go to: https://www.kaggle.com/settings"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Save kaggle.json to this directory"
    echo "  4. Run this script again"
    echo ""
    exit 1
fi

echo "✓ Found kaggle.json"
echo ""

# Step 1: Create virtual environment
echo "Step 1: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo "✓ Virtual environment ready"
echo ""

# Step 2: Install packages
echo "Step 2: Installing Python packages..."
echo "This will take 5-10 minutes..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✓ Packages installed"
echo ""

# Step 3: Setup Kaggle and download data
echo "Step 3: Downloading data from Kaggle..."
echo "This will take 5-10 minutes..."
python src/data_pipeline/setup_kaggle.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Data download failed!"
    echo ""
    echo "Common issues:"
    echo "  1. Need to accept dataset licenses on Kaggle"
    echo "     Visit: https://www.kaggle.com/datasets/nphantawee/pump-sensor-data"
    echo "     Visit: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product"
    echo "     Click 'Download' on each to accept the license"
    echo ""
    echo "  2. Check your kaggle.json credentials"
    echo "  3. Check internet connection"
    echo ""
    exit 1
fi

echo ""

# Step 4: Generate mock text
echo "Step 4: Generating mock text data..."
python src/data_pipeline/generate_mock_text.py
echo "✓ Mock text generated"
echo ""

# Done
echo "========================================="
echo "✓✓✓ SETUP COMPLETE! ✓✓✓"
echo "========================================="
echo ""
echo "Data downloaded and ready!"
echo ""
echo "Next steps:"
echo "  1. Preprocess: ./run_preprocessing.sh"
echo "  2. Train:      ./run_training.sh"
echo "  3. Start API:  ./run_api.sh"
echo ""
echo "Or run everything: ./run_all.sh"
echo ""
