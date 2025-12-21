Topic       : Courier Assignment Optimization using Deep Q-Networks (DQN)
Author      : Beiqi Zhou
Date        : December 21, 2025

# Background
================================================================================
## Background
================================================================================

This project implements a **real-time courier assignment system** using **Deep Reinforcement Learning (DQN)**.
The goal is to **minimize delivery cost** while respecting courier availability, distance, and workload.

Key challenges:
- Couriers move continuously; assignment decisions must be made in real time.
- Historical data shows suboptimal assignments (reassignments, high cost).
- Need a model that learns **cost-aware, load-balanced** routing decisions.

We use **DQN** to learn a policy that selects the best courier from a small pool of nearby candidates at the moment an order is created.

The system is split into:
1. **Training Pipeline** (`interview_test_final.py`) — trains DQN and saves model + test episodes.
2. **Inference-Only Evaluation** (`interview_test_final_2.py`) — simulates real-time assignment using saved model.

---

================================================================================
## Details of Functions Performed in This Project
================================================================================

### 1. `src/util/DataLoaders.py`
- `FileDataLoader`: Loads `final_dataset.csv` into a Pandas DataFrame.

### 2. `src/util/DataPreprocessing.py`
- Filters data to 24-hour window.
- Computes Haversine distance from courier → restaurant.
- Adds `cost = distance_km × RATE_PER_KM`.
- Tracks `open_orders` per courier.
- Generates **synthetic negative candidates** (nearby couriers not chosen).
- Outputs:
  - `feat_df`: Real (order, courier) attempts with metadata.
  - `feat_df_new`: Real + synthetic candidates.
  - `df_filtered`: Full filtered tracking data.

All config parameters (`RATE_PER_KM`, `TIME_WINDOW_MIN`, `N_CANDIDATES`) are loaded from JSON.

### 3. `src/util/Predictors.py`
- `DQNCourierPredictor`: Inherits from abstract `Model`.
- Normalizes features: `cost`, `open_orders`, `hour`.
- Builds **episodes** (one per order) with states, actions, rewards.
- Implements:
  - DQN network (3 → 64 → 64 → 5)
  - Replay buffer
  - ε-greedy exploration
  - Target network updates
- **Trains for 15 epochs**.
- **Saves**:
  - `dqn_courier_model.pth` (model + normalization stats)
  - `test_episodes.pkl` (held-out test set)

### 4. `src/util/RealTimeEvaluation.py`
- **Inference-only** module.
- Loads:
  - `dqn_courier_model.pth`
  - `test_episodes.pkl`
- For each test order:
  - Waits `delta_minutes` (default: 5)
  - Finds **top-3 closest available couriers**
  - Builds normalized state
  - Uses **DQN to pick best courier**
- Compares **predicted cost** vs **historical cost**
- Reports **% cost reduction**

### 5. `src/interview_test_final.py`
- Full training + evaluation pipeline.
- Runs preprocessing → training → saves model & test episodes.

### 6. `src/interview_test_final_2.py`
- **Real-time simulation only**.
- No training.
- Fast inference using saved model and episodes.

### 7. `resources/interview-test-final.json`
```json
{
  "RATE_PER_KM": 0.2,
  "TIME_WINDOW_MIN": 5,
  "N_CANDIDATES": 2,
  "MODEL_SAVE_PATH": "dqn_courier_model.pth",
  "MODEL_LOAD_PATH": "dqn_courier_model.pth",
  "TEST_EPISODES_PATH": "test_episodes.pkl"
}

================================================================================
## Sample Output of Performance Metrics
================================================================================

2025-10-29 18:42:48;INFO;Epoch  1 | Train reward: +43.18
2025-10-29 18:42:49;INFO;Epoch  2 | Train reward: +82.72
2025-10-29 18:42:50;INFO;Epoch  3 | Train reward: +99.92
2025-10-29 18:42:51;INFO;Epoch  4 | Train reward: +101.73
2025-10-29 18:42:52;INFO;Epoch  5 | Train reward: +111.85
2025-10-29 18:42:53;INFO;Epoch  6 | Train reward: +119.82
2025-10-29 18:42:54;INFO;Epoch  7 | Train reward: +124.68
2025-10-29 18:42:56;INFO;Epoch  8 | Train reward: +130.17
2025-10-29 18:42:57;INFO;Epoch  9 | Train reward: +140.28
2025-10-29 18:42:58;INFO;Epoch 10 | Train reward: +151.86
2025-10-29 18:42:59;INFO;Epoch 11 | Train reward: +153.16
2025-10-29 18:43:01;INFO;Epoch 12 | Train reward: +148.76
2025-10-29 18:43:02;INFO;Epoch 13 | Train reward: +149.98
2025-10-29 18:43:03;INFO;Epoch 14 | Train reward: +159.46
2025-10-29 18:43:04;INFO;Epoch 15 | Train reward: +151.37

================================================================
REAL-TIME COURIER ASSIGNMENT SIMULATION
================================================================
Orders processed:          257
Model total cost:     $      8.28
Historical cost:      $     71.83
COST REDUCTION:          88.48%
================================================================






