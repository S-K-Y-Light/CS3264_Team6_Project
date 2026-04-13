# CS3264_Team6_Project
## Autonomous Vehicle Pedestrian Temporal Risk Evaluation

This repository (**currently under the feature/temporal-risk-model branch**) contains a modular machine learning pipeline designed to predict pedestrian crossing intent for autonomous vehicle safety. Moving beyond static frame-by-frame classification, this system implements **Recursive Bayesian Updating** to track pedestrian hesitation, evaluate temporal risk, and trigger distance-weighted emergency alerts.

---

## 📦 Prerequisites & Setup

Before running the notebooks, ensure your local environment is correctly configured:

- **JAAD Dataset:** Make sure you have the JAAD Dataset locally. It is highly recommended to clone the [JAAD Dataset 2.0](https://github.com/ykotseruba/JAAD) before working on the project. Ensure your local paths in the notebooks point to this directory.

- **Pre-computed Pose Data:** To save computation time on pose extraction, download the `pedestrian_poses.pkl` file from the link below and place it in the **root directory** of this repository.

  📥 [Download pedestrian_poses.pkl from Google Drive](https://drive.google.com/file/d/19aaBSTmzji-Jw5PXO0YQYSyZ61-KnzrC/view?usp=drive_link)

---

## ⚙️ Workflow & Execution Order

To reproduce the environment and generate the final temporal risk evaluation metrics, execute the notebooks in the following sequential order:

| Step | Notebook | Role |
|------|----------|------|
| 1 | `main.ipynb` | Data Extraction & Baselines |
| 2 | `pedestrian_poses.ipynb` | Pose & Kinematics Feature Engineering |
| 3 | `train_advanced_model.ipynb` | Advanced ML Frame Classification |
| 4 | `temporal_risk_evluation.ipynb` | Temporal Bayesian Safety Engine |

---

## 📁 Notebook Directory & Functions

### 1. `main.ipynb` & `main2.ipynb` — Data Extraction & Baselines
- **Function:** Ingests the raw JAAD dataset, extracts bounding box geometries, formats the ground truth labels, and extracts raw image frames.
- **Output:** Trains the baseline XGBoost classifier and outputs `baseline_predictions.csv` for downstream performance comparison.

### 2. `pedestrian_poses.ipynb` — Feature Engineering
- **Function:** Extracts structural and kinematic data by passing JAAD video frames through a pre-trained YOLO26 model.
  > **Note:** If you downloaded `pedestrian_poses.pkl` in the setup steps, you can skip running the heavy inference in this notebook.
- **Output:** Generates `pedestrian_poses.pkl`, containing full-frame coordinates and mapped human joint keypoints to detect subtle physical cues (e.g., leaning, leg angles).

### 3. `train_advanced_model.ipynb` — Core Frame Classification
- **Function:** Merges the baseline data with the newly extracted pose kinematics. Splits data strictly by `pedestrian_id` using `GroupShuffleSplit` to prevent temporal data leakage.
- **Mathematical Logic:**
  - **Isotonic Calibration:** Random Forests suffer from probability shrinkage (pushing outputs away from 0 and 1). Isotonic calibration is applied via `CalibratedClassifierCV` to map raw tree outputs into true, reliable probability confidence scores required for the downstream Bayesian model.
- **Output:** `advanced_predictions.csv`

### 4. `temporal_risk_evluation.ipynb` — The Safety Engine
- **Function:** The core autonomous safety layer. Ingests noisy, frame-by-frame probabilities and applies a Bayesian filter to calculate a smooth, accurate risk score over time, triggering alerts based on dynamic thresholds.
- **Output:** `temporal_evaluation_results.csv` and final metrics.
  - ✅ **Temporal AUROC: 0.9759**
  - ✅ **Average Warning Lead Time: 2.14s**
  - ✅ **True Model Misses: 0.6%**

---

## 🧮 Mathematical Logic: The Temporal Bayesian Engine

Instead of a simple rolling average (which introduces dangerous lag), the final notebook uses real-time probabilistic tracking to handle mixed, noisy predictions across sequential frames.

### Log-Odds Conversion for Numerical Stability
Probabilities are transformed into log-odds to prevent mathematical underflow when multiplying small numbers. The Bayesian update rule becomes simple addition:

$$\text{LogOdds}_{new} = \text{LogOdds}_{prior} + \text{LogOdds}_{observation}$$

### The Forgetting Factor (Dynamic Environments)
Pedestrians change their minds. To prevent the model from getting "stuck" on an old belief when a pedestrian hesitates or retreats, a forgetting factor decays the prior state towards maximum uncertainty (0.5) by 5% at the start of every frame:

$$P_{prior} = 0.5 + (P_{prior} - 0.5) \times 0.95$$

### Asymmetric Uncertainty Decay
Raw observations are penalized by confidence. Predictions leaning heavily towards "crossing" or "not crossing" are mathematically shrunk towards 0.5, ensuring the system relies on **sustained, high-confidence sequences** rather than single-frame spikes — without artificially biasing the model towards absolute certainty.

### Distance-Weighted Dynamic Threshold
Bounding box height serves as a proxy for pedestrian depth/distance:

| Pedestrian Distance | Bbox Height | Threshold | Behaviour |
|---|---|---|---|
| Far | `< 10%` | `0.60` | Strict — prevent false braking |
| Mid-range | `10–20%` | `0.30` | Balanced |
| Close | `> 20%` | `0.35` | Aggressive — prioritise safety |
| Emergency | `> 25%` + RF prob `≥ 0.30` | Instant trigger | Bypass trend window entirely |