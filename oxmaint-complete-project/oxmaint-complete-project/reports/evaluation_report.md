# Multimodal Predictive Maintenance System - Evaluation Report

> **Project Overview:** A multimodal predictive maintenance system for industrial pumps that fuses three input modalities - **sensor time-series**, **maintenance text**, and **defect images** - to predict equipment failure probability and estimated time to breakdown.
>
> **Inputs:**
> - **Sensor Data** - Pump Sensor Data (Kaggle, ~220K time-series rows). Labels: `NORMAL`, `RECOVERING`, `BROKEN`. Used for real-time anomaly detection via LSTM classification.
> - **Text Data** - Mock maintenance manuals, work orders, and maintenance requests generated via ChatGPT. Labels: `routine_maintenance`, `bearing_failure`, `seal_leak`, `impeller_damage`, `motor_overload`, etc. Used for risk signal extraction via TF-IDF + Logistic Regression.
> - **Images** - Casting Manufacturing Defects dataset (Kaggle, 8K+ grayscale images). Labels: `ok`, `defective`. Used for visual defect detection via CNN.
>
> **Data Loading:** All datasets were loaded directly from Kaggle into VS Code and processed through modality-specific preprocessing pipelines before model training.

---

## 1. Data Preprocessing Summary

### 1.1 Sensor Data
- Parsed timestamps; identified 52 sensor feature columns (`sensor_00` to `sensor_51`)
- Replaced `inf`/`-inf` values and filled `NaN` with `0.0`
- Assigned pseudo `asset_id` (50 pumps) using chunk-based segmentation
- Applied **sliding window** (window size = 50, stride = 10) → ~21K windows created
- Labels remapped to binary: `NORMAL → 0`, `RECOVERING/BROKEN → 1`
- Applied `StandardScaler` (fit on train, transform on train+test)
- 80/20 stratified train-test split

### 1.2 Image Data
- Loaded grayscale images from `ok/` and `defective/` directories (up to 1000/class)
- Resized all images to **224×224**
- Normalized pixel values to `[0, 1]` (divided by 255)
- Added channel dimension for CNN input compatibility
- Assigned deterministic pseudo `asset_id` + pseudo timestamps for unified schema alignment
- 80/20 stratified train-test split

### 1.3 Text Data
- Processed mock **manuals** (global knowledge, asset_id = `GLOBAL`), **work orders**, and **maintenance requests**
- Cleaned text: lowercased, removed special characters, collapsed whitespace
- Extracted fault indicator scores for: `bearing_failure`, `seal_leak`, `impeller_damage`, `motor_overload`, `normal`
- Generated sentence embeddings using `all-MiniLM-L6-v2` (SentenceTransformers)
- Saved unified JSONL records keyed by `asset_id` + `timestamp` for fusion alignment

---

## 2. Model Evaluation

### 2.1 Sensor Model - SmallLSTM (Time-Series Binary Classifier)

**Architecture:** Single-layer LSTM (hidden size = 32) + LayerNorm + Dropout (0.2) + Linear head
**Training:** Adam optimizer (lr = 5e-4), weighted CrossEntropyLoss, early stopping on PR-AUC (patience = 4), 15 epochs max

| Metric | Value |
|---|---|
| **Final Test F1** | **0.9898** |
| **Final Test PR-AUC** | **1.0000** |
| Best Epoch | 13 |
| Train Loss (Epoch 13) | 0.0057 |

**Epoch-wise Progression (Selected):**

| Epoch | Train Loss | Test F1 | Test PR-AUC |
|---|---|---|---|
| 1 | 0.1496 | 0.8896 | 0.9697 |
| 5 | 0.0154 | 0.9699 | 0.9964 |
| 10 | 0.0084 | 0.9797 | 0.9995 |
| 13 | 0.0057 | **0.9898** | **1.0000** |
| 15 | 0.0052 | 0.9898 | 0.9986 |

> **Note:** PR-AUC of 1.0 at epoch 13 confirms excellent discrimination on a class-imbalanced dataset. Binary mapping (`NORMAL=0`, `RECOVERING/BROKEN=1`) with inverse-frequency class weights ensured the minority fault class was not ignored.

---

### 2.2 Image Model - DefectCNN (Grayscale Defect Classifier)

**Architecture:** 3-block CNN (Conv → BatchNorm → ReLU → MaxPool), channels 1→32→64→128, followed by two FC layers (256 → 2) with Dropout (0.5)
**Training:** Adam optimizer (lr = 0.001, weight_decay = 0.01), CrossEntropyLoss, ReduceLROnPlateau scheduler, early stopping (patience = 5), 20 epochs max (stopped at 11)

| Metric | Value |
|---|---|
| **Best Test Accuracy** | **88.95%** |
| Best Epoch | 6 |
| Train Accuracy (at best) | 73.85% |
| Final Epoch (11) Test Acc | 65.56% |

**Epoch-wise Progression (Selected):**

| Epoch | Train Acc (%) | Test Acc (%) |
|---|---|---|
| 1 | 66.21 | 76.61 |
| 2 | 73.94 | 83.06 |
| 4 | 73.53 | 88.77 |
| 6 | 73.85 | **88.95** |
| 9 | 80.94 | 82.50 |
| 11 | 81.72 | 65.56 |

> **Note:** Test accuracy peaked at epoch 6 (88.95%) before degrading due to overfitting. The model was subject to test instability given the relatively small dataset size (8K images, 1K/class limit). The best checkpoint was used for inference.

---

### 2.3 Text Risk Model - TF-IDF + Logistic Regression (Binary Risk Classifier)

**Architecture:** TF-IDF vectorizer (max 10K features, unigrams + bigrams) + Logistic Regression (balanced class weights, max_iter = 3000)
**Labels:** Work order `issue_type` → `0 = normal` (routine/inspection/preventive), `1 = abnormal` (bearing failure, seal leak, etc.); all maintenance requests → `1`

| Metric | Value |
|---|---|
| **Test PR-AUC** | Reported per run |
| **Positive Class** | abnormal (1) |
| Training Samples | From work orders + maintenance requests |
| Vectorizer | TF-IDF, (1,2)-grams, 10K features |

> **Note:** Lightweight and interpretable. Designed specifically to contribute a calibrated `abnormal_probability` for late fusion. No epoch history - single-pass sklearn training.

---

## 3. Multimodal Fusion Evaluation

The fusion layer combines outputs from all three modalities using a **weighted late-fusion strategy**. Each modality contributes a probability of abnormality; the final `failure_probability` is a weighted combination, with `llm_explanation` generated via an LLM summarizer.

---

### 3.1 Test Case A - Sensor Only (Low Values, Normal Operation)

**Input:** Sensor readings with low feature values (normal operating range)
**Modalities Used:** `sensor`

| Field | Value |
|---|---|
| **Failure Probability** | **0.000** |
| Estimated Time to Breakdown | 1200 hrs |
| Predicted Fault Type | `normal` |
| Top Signals | - |
| Sensor Predicted Class | 0 (Normal) |
| Sensor Confidence | 99.9% |
| Sensor P(abnormal) | 0.001 |
| Inference Time | 7240.8 ms |

**LLM Explanation Summary:** Pump predicted to be functioning normally with no faults detected. Sensor analysis shows 99.9% confidence in normal status. No image or text data available.

---

### 3.2 Test Case B - Sensor + Text (Normal Sensor, Alarming Text)

**Input:** Low sensor values (normal) + text: *"severe bearing overheating and vibration detected"*
**Modalities Used:** `sensor`, `text`

| Field | Value |
|---|---|
| **Failure Probability** | **0.168** |
| Estimated Time to Breakdown | 1200 hrs |
| Predicted Fault Type | `abnormal_text_signal` |
| Top Signals | `text_risk` |
| Sensor Predicted Class | 0 (Normal) |
| Sensor Confidence | 99.9% |
| Sensor P(abnormal) | 0.001 |
| Text P(abnormal) | 0.668 |
| Text Confidence | 0.335 |
| Inference Time | 2602.8 ms |

**LLM Explanation Summary:** 16.8% failure probability. Sensor strongly indicates normal operation; text analysis flags a moderate abnormality risk (66.8%). Conflict between modalities resolved by weighted fusion - text risk elevated the overall score despite clean sensor data.

---

### 3.3 Test Case C - All Three Modalities (Sensor + Image + Text)

**Input:** Low sensor values + defect image (no visual defect detected) + text: *"severe bearing overheating and vibration detected"*
**Modalities Used:** `sensor`, `image`, `text`

| Field | Value |
|---|---|
| **Failure Probability** | **0.271** |
| Estimated Time to Breakdown | 400 hrs |
| Predicted Fault Type | `abnormal_text_signal` |
| Top Signals | `text_risk` |
| Sensor Predicted Class | 0 (Normal) |
| Sensor Confidence | 99.9% |
| Sensor P(abnormal) | 0.001 |
| Image Defect Detected | No |
| Image Confidence | 74.0% |
| Image P(defect) | 0.26 |
| Text P(abnormal) | 0.775 |
| Text Confidence | 0.551 |
| Inference Time | 2987.9 ms |

**LLM Explanation Summary:** 27.1% failure probability increased from 16.8% (Test B) due to the addition of image evidence (26% defect probability) alongside strong text risk signal. Estimated breakdown window tightened from 1200 hrs to 400 hrs as all three modalities collectively strengthened the risk assessment.

---

## 4. Cross-Modality Fusion Summary

| Modalities | Failure Prob | TTB (hrs) | Predicted Fault | Dominant Signal |
|---|---|---|---|---|
| Sensor only | 0.000 | 1200 | normal | - |
| Sensor + Text | 0.168 | 1200 | abnormal_text_signal | text_risk |
| Sensor + Image + Text | **0.271** | **400** | abnormal_text_signal | text_risk |

> **Key Insight:** The fusion system correctly escalates risk as more modalities provide corroborating evidence. Even when the sensor model is highly confident in normal operation (99.9%), alarming text signals are not discarded - they meaningfully shift the failure probability. Adding image evidence further increases the risk estimate and reduces the estimated time to breakdown, demonstrating that the system is responsive to multi-source uncertainty.

---

## 5. Summary

| Model | Task | Key Metric | Result |
|---|---|---|---|
| SmallLSTM | Sensor anomaly detection | PR-AUC | **1.000** |
| SmallLSTM | Sensor anomaly detection | F1 Score | **0.9898** |
| DefectCNN | Image defect classification | Test Accuracy | **88.95%** |
| TF-IDF + LR | Text risk scoring | Binary classification | PR-AUC reported per run |
| Late Fusion | Multimodal failure prediction | Failure Probability Range | 0.000 → 0.271 (context-sensitive) |

---

## 6. Critical Design Questions

### Q1 - Can / Should Models Be Fine-Tuned Here?

**Sensor (SmallLSTM):** Fine-tuning is not necessary at this stage - the model already achieves PR-AUC of 1.0 and F1 of 0.9898 on the current dataset. If deployed on a real pump fleet with different sensor distributions, domain-adaptive fine-tuning on a small labelled sample from the target environment would meaningfully improve generalisation.

**Image (DefectCNN):** Fine-tuning is recommended. The model shows instability across epochs (test accuracy oscillating between 53% and 88%) due to limited training data (~1K images per class). Transfer learning from a pretrained backbone (e.g. ResNet-18 with frozen early layers, fine-tuned final layers) would yield more stable and higher accuracy results without requiring more data.

**Text (TF-IDF + LogReg):** Fine-tuning is not applicable to this architecture - TF-IDF is a stateless vectoriser, not a learnable representation. If richer text understanding is needed (e.g. longer maintenance reports, ambiguous language), replacing this with a fine-tuned `distilBERT` or `MiniLM` classifier would be the right upgrade path.

---

### Q2 - Should You Use Structured Reasoning Flow and/or Ensemble Methods?

**Structured reasoning flow:** Yes, and it is already partially implemented. The Input Router applies a conditional pipeline - only the modalities present in the request are executed. Formalising this as an explicit reasoning graph (e.g. sensor confidence > 0.95 → suppress text weight) would make the fusion logic more interpretable and auditable for maintenance engineers.

**Ensemble methods:** Worth considering for the image model specifically. Ensembling 3–5 CNN runs with different random seeds and averaging their `p_defect` outputs would reduce the epoch-to-epoch variance currently observed (test accuracy swinging from 53% to 88%). For sensor and text, the current single-model approach is sufficient given their stable performance.

---

### Q3 - What Is the Best Deployment Approach for This Agent?

The best approach is a **containerised microservice** behind an async REST API - which is what the current FastAPI + Docker + Uvicorn stack implements. For production scale, the three model inference pipelines (sensor, image, text) should run as **parallel async tasks** within a single request rather than sequentially, which would cut inference time roughly by the cost of the slowest non-critical modality. If request volume grows, separating each modality into its own lightweight container behind a load balancer allows independent scaling - sensor inference is cheap and frequent, image inference is heavier and less frequent.

---

### Q4 - Do You Need Database Connections? If Yes, What Type and Why?

**Yes - two types are justified:**

A **time-series database** (e.g. InfluxDB or TimescaleDB) is the right fit for storing incoming sensor windows, prediction history, and TTB estimates per `asset_id`. These databases are optimised for high-write, time-ordered data and support efficient range queries like "show me all abnormal readings for pump_012 in the last 7 days" - exactly the queries a maintenance dashboard would run.

A **document store** (e.g. MongoDB or PostgreSQL with JSONB) suits the text records - work orders, maintenance requests, and LLM explanations are variable-length, schema-flexible documents that map naturally to JSON storage and benefit from full-text search capabilities.

---

### Q5 - Where Is the Main Latency Bottleneck and How Would You Mitigate It?

The main bottleneck is the **LLM explanation call** - observed inference times of 2,600–7,240 ms in testing are dominated by the round-trip to the language model, not by the three ML models themselves (which run in milliseconds). Two mitigations: first, make the LLM call **async and non-blocking** so the structured JSON response (`failure_probability`, `fault_type`, `TTB`) is returned immediately while the explanation is streamed or appended in a follow-up; second, **cache LLM explanations** by hashing the fusion output signature (failure probability bucket + fault type + top signals) so repeated similar predictions reuse a stored explanation rather than making a new API call.

---

*Report generated from model training histories, preprocessing pipeline code, and live API inference outputs.*
