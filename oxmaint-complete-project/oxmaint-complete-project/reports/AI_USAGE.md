# AI Usage Documentation

This document details how generative AI tools were used as assistants during the development of the Multimodal Predictive Maintenance System.

## Overview

The core idea, system architecture, technology decisions, dataset selection, model choices, and overall project direction were conceived and driven entirely by me. Generative AI tools — primarily **ChatGPT**, **Claude**, and **Gemini** — were used as coding assistants during implementation, similar to how a developer might use Stack Overflow or documentation, but interactively.

**AI tools were used for:**
- Generating boilerplate and repetitive code blocks based on my specifications
- Suggesting syntax and implementation patterns I described in plain language
- Helping debug specific errors during testing
- Drafting initial documentation structure which was then rewritten

**AI tools were NOT responsible for:**
- The project idea or problem framing
- Dataset selection and justification
- Architecture design and component decisions
- Model selection and justification
- Fusion strategy design
- Testing, validation, and result interpretation
- Any final decisions on what stays in the codebase

---

## My Contributions vs. AI Assistance

### Architecture & Design — Entirely Mine

I designed the full system architecture: the three-modality pipeline (sensor → LSTM, image → CNN, text → TF-IDF+LR), the late fusion strategy, the FastAPI orchestration layer, and the deployment setup. I chose to use a **weighted late fusion** approach after evaluating alternatives (early fusion, joint training) and determining that independent per-modality models would give cleaner failure signals and graceful degradation when modalities are missing.

Technology choices were entirely my own:
- **PyTorch** — chosen for flexibility in custom training loops and debugging
- **FastAPI** — chosen over Flask for native async support and automatic Swagger docs
- **SentenceTransformers (MiniLM)** — chosen for lightweight, fast text embeddings without requiring a GPU
- **Docker + Uvicorn** — chosen for reproducible deployment

AI tools played no role in these decisions.

---

### Dataset Selection — Entirely Mine

I independently identified and selected:
- **Pump Sensor Data** (Kaggle, ~220K rows) for time-series anomaly detection
- **Casting Manufacturing Defects** (Kaggle, 8K+ images) for visual defect detection
- **ChatGPT-generated mock text** for maintenance work orders and manuals — this was my own idea to simulate realistic CMMS records when no labelled text dataset existed

---

### Model Selection — Entirely Mine

I chose each model based on the constraints of the data and the deployment target:

- **SmallLSTM** — I specifically wanted a lightweight LSTM (not a Transformer) because the sensor windows are short (50 timesteps) and the system needs to run on CPU. I brainstormed the architecture (hidden=32, single layer, binary head) and asked AI tools to help write the training loop boilerplate around it.
- **DefectCNN** — I specified a 3-block convolutional architecture with BatchNorm and Dropout. I asked AI to help structure the PyTorch `nn.Module` class from my description.
- **TF-IDF + Logistic Regression** — My deliberate choice to keep the text model interpretable and fast, rather than a fine-tuned BERT which would be overkill for short maintenance phrases.

---

### Code Generation — AI-Assisted, Human-Verified

For the implementation of components I had already designed, I used AI tools to help write code faster. My workflow was:

1. I described what I needed in plain language (e.g. "write a sliding window function over a pandas dataframe with configurable size and stride, output numpy arrays")
2. AI generated an initial version
3. I tested it, identified issues, and either fixed them myself or asked for specific changes
4. I validated outputs at every step before moving on

**Verified by running:**
```bash
# Preprocessing validation
python src/preprocessing/sensor_preprocessing.py
python src/preprocessing/image_preprocessing.py
python src/preprocessing/text_preprocessing.py

# Shape and dtype checks
python -c "
import numpy as np
X = np.load('data/processed/sensor/X_train.npy')
print(f'Shape: {X.shape}, Type: {X.dtype}')
"

# Model training validation
python src/modeling/train_sensor_model.py
cat models/sensor_model/training_history.json

# API smoke test
uvicorn src.api.main:app --reload &
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"asset_id": "pump_001", "sensor_window": [[...]]}'
```

**Issues found during testing (fixed by me):**
- Sensor preprocessing: label remapping logic was incorrect for multi-class → binary conversion; I rewrote the `remap_labels_to_binary` function myself
- Image preprocessing: path discovery failed for nested dataset folder structures; I debugged and added the recursive search logic
- Text preprocessing: embedding model loading was blocking the API startup; I restructured it to lazy-load on first inference call
- Fusion weights: initial equal weighting produced poor results when modalities conflicted; I adjusted weights based on observing the test case outputs

---

### Fusion Logic — Entirely Mine

The fusion strategy — weighted combination of per-modality probabilities, dynamic weight adjustment when modalities are absent, TTB heuristic thresholds, and routing to an LLM for explanation — was designed entirely by me. I asked AI tools to help implement the Python class structure once I had the logic mapped out.

---

### Testing & Evaluation — Entirely Mine

All test cases were designed by me. I tested the API with three deliberate scenarios:
- Sensor only with normal readings — to verify the baseline path
- Sensor (normal) + alarming text — to verify the fusion correctly elevates risk from text signals even when sensor is clean
- All three modalities — to verify additive risk escalation

I interpreted the results, identified what the numbers meant, and wrote the evaluation report.

---

## Where AI Tools Were Most Useful

| Task | Tool Used | My Role |
|---|---|---|
| Training loop boilerplate (PyTorch) | ChatGPT, Claude | Specified architecture, hyperparameters, and loss function |
| FastAPI route structure | Claude | Specified endpoints, request/response schema |
| Dockerfile and docker-compose | ChatGPT | Specified base image, dependencies, volume mounts |
| Initial README draft | Claude, Gemini | Rewrote and customised entirely |
| Debugging specific error messages | ChatGPT | Identified root cause myself, used AI for syntax help |
| Shell script structure | Claude | Specified pipeline order and error handling requirements |

---

## Honest Assessment

AI tools accelerated the coding phase significantly. Without them, writing the same volume of boilerplate (dataset classes, training loops, API schemas, Docker config) would have taken considerably longer. However, every line that went into the final codebase was:

- **Read and understood by me**
- **Tested against real data**
- **Modified where it did not match my design intent**
- **Validated for correctness of output**

The project would not exist in its current form without the AI tools helping with implementation speed. But the tools did not make a single design decision — they implemented decisions I had already made.

---

*The system architecture, model selection, fusion strategy, dataset choices, evaluation methodology, and all final decisions are my own work. AI tools were used as implementation accelerators under continuous human oversight.*
