# Courier Assignment Optimization using Deep Q-Networks (DQN)

**Author:** Beiqi Zhou  
**Date:** December 21, 2025

---

## Background

This project implements a **real-time courier assignment system** using **Deep Reinforcement Learning (DQN)**.  
The goal is to **minimize delivery cost** while respecting courier availability, distance, and workload.

### Key Challenges
- Couriers move continuously; assignment decisions must be made in real time.
- Historical data shows suboptimal assignments (reassignments, high cost).
- A model is needed that learns **cost-aware, load-balanced** routing decisions.

We use **DQN** to learn a policy that selects the best courier from a small pool of nearby candidates at the moment an order is created.

### System Overview
The system is split into two pipelines:

1. **Training Pipeline** (`interview_test_final.py`)  
   Trains the DQN model and saves the trained model and test episodes.

2. **Inference-Only Evaluation** (`interview_test_final_2.py`)  
   Simulates real-time courier assignment using the saved model.

---

## Project Structure & Functional Details

### 1. `src/util/DataLoaders.py`
- **`FileDataLoader`**
  - Loads `final_dataset.csv` into a Pandas DataFrame.

---

### 2. `src/util/DataPreprocessing.py`
- Filters data to a 24-hour window.
- Computes Haversine distance from courier → restaurant.
- Adds cost calculation:


- Tracks `open_orders` per courier.
- Generates **synthetic negative candidates** (nearby couriers not chosen).
- Outputs:
- `feat_df`: Real (order, courier) attempts with metadata.
- `feat_df_new`: Real + synthetic candidates.
- `df_filtered`: Filtered tracking data.

All parameters (`RATE_PER_KM`, `TIME_WINDOW_MIN`, `N_CANDIDATES`) are loaded from JSON config.

---

### 3. `src/util/Predictors.py`
- **`DQNCourierPredictor`**
- Inherits from abstract `Model`.
- Normalizes features:
  - `cost`
  - `open_orders`
  - `hour`
- Builds **episodes** (one per order) with states, actions, and rewards.
- Implements:
  - DQN network: `3 → 64 → 64 → 5`
  - Replay buffer
  - ε-greedy exploration
  - Target network updates
- Trains for **15 epochs**.

**Saved artifacts:**
- `dqn_courier_model.pth` (model + normalization stats)
- `test_episodes.pkl` (held-out test set)

---

### 4. `src/util/RealTimeEvaluation.py`
- **Inference-only module**
- Loads:
- `dqn_courier_model.pth`
- `test_episodes.pkl`
- For each test order:
- Waits `delta_minutes` (default: 5)
- Finds **top-3 closest available couriers**
- Builds normalized state
- Uses **DQN** to select the best courier
- Compares **predicted cost** vs **historical cost**
- Reports **percentage cost reduction**

---

### 7. Configuration (`resources/interview-test-final.json`)
```json
{
"RATE_PER_KM": 0.2,
"TIME_WINDOW_MIN": 5,
"N_CANDIDATES": 2,
"MODEL_SAVE_PATH": "dqn_courier_model.pth",
"MODEL_LOAD_PATH": "dqn_courier_model.pth",
"TEST_EPISODES_PATH": "test_episodes.pkl"
}





