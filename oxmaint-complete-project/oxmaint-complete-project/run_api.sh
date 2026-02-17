#!/bin/bash
# Start the FastAPI server

# Activate venv
source venv/bin/activate

echo "Starting Oxmaint Predictive Maintenance API..."
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""

# Start server with uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
