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






