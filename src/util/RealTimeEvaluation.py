"""
RealTimeEvaluation.py
=====================
Real-time courier assignment simulation using a pre-trained DQN model.

Features:
- Loads test_episodes.pkl (saved during training)
- Loads dqn_courier_model.pth
- Uses config for paths and params
- No training — inference only
- Fast, clean, reusable
"""

import logging
import os
import pickle
from typing import List, Dict, Tuple, Any
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import timedelta
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

# =============================================================================
# 1. CONFIG-DRIVEN PATHS & GLOBAL SETTINGS
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2. DQN MODEL DEFINITION (must match training)
# =============================================================================
class DQN(nn.Module):
    """Exact same architecture as used in training."""

    def __init__(self, state_dim: int = 3, n_actions: int = 5,
                 hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# 3. REAL-TIME EVALUATOR CLASS
# =============================================================================
class RealTimeEvaluator:
    """Loads model and performs inference on candidate couriers."""

    def __init__(self, model_path: str, norm_stats: Dict[str, float]):
        self.device = DEVICE
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load model
        self.model = DQN(
            state_dim=3,
            n_actions=checkpoint.get('max_actions', 5),
            hidden=64
        ).to(self.device)
        self.model.load_state_dict(checkpoint['policy_net'])
        self.model.eval()

        # Normalization stats
        self.norm = norm_stats

        logging.info(f"Model loaded from {model_path} on {self.device}")

    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float,
                     lon2: float) -> float:
        """Haversine distance in kilometers."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    def _normalize(self, cost: float, open_orders: int,
                   hour: int) -> np.ndarray:
        return np.array([
            (cost - self.norm['cost_mean']) / (self.norm['cost_std'] + 1e-8),
            (open_orders - self.norm['open_mean']) / (
                        self.norm['open_std'] + 1e-8),
            hour / 23.0
        ], dtype=np.float32)

    def select_best_courier(self,
                            candidates: List[
                                Tuple[int, float, int, float, float]],
                            eval_hour: int) -> Tuple[int, float]:
        """
        Select best courier from candidates using DQN.

        Args:
            candidates: List of (courier_id, dist_km, open_orders, c_lat, c_lon)
            eval_hour: Hour of day for state

        Returns:
            (best_courier_id, predicted_cost)
        """
        if not candidates:
            return -1, float('inf')

        states = []
        costs = []
        for cid, dist_km, open_orders, _, _ in candidates:
            cost = 0.2 * dist_km  # RATE_PER_KM
            state = self._normalize(cost, open_orders, eval_hour)
            states.append(state)
            costs.append(cost)

        states_tensor = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            q_values = self.model(states_tensor)
            best_idx = q_values[:,
                       0].argmax().item()  # Use first action as "select"

        return candidates[best_idx][0], costs[best_idx]


# =============================================================================
# 4. DATA PREPARATION HELPERS
# =============================================================================
def add_open_orders_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'open_orders' column to courier tracking data."""

    def count_open(courier_id: str, ts: pd.Timestamp) -> int:
        return len(df[
                       (df['courier_id'] == courier_id) &
                       (df['courier_location_timestamp'] <= ts)
                       ])

    df = df.copy()
    df['open_orders'] = df.apply(
        lambda row: count_open(row['courier_id'],
                               row['courier_location_timestamp']),
        axis=1
    )
    return df


def build_new_orders_from_episodes(test_episodes: List[Dict],
                                   feat_df: pd.DataFrame) -> pd.DataFrame:
    """Extract order metadata from test episodes."""
    order_numbers = [ep['order_number'] for ep in test_episodes]

    info_cols = ['order_number', 'order_created_timestamp', 'restaurant_lat',
                 'restaurant_lon']
    order_info = feat_df.drop_duplicates('order_number')[info_cols]

    new_orders = pd.DataFrame({'order_number': order_numbers})
    return new_orders.merge(order_info, on='order_number', how='left')


# =============================================================================
# 5. MAIN EVALUATION LOOP
# =============================================================================
def run_evaluation(evaluator: RealTimeEvaluator,
                   new_orders_df: pd.DataFrame,
                   courier_df: pd.DataFrame,
                   delta_minutes: int) -> pd.DataFrame:
    """Run real-time assignment simulation."""
    results = []

    for _, order in tqdm(new_orders_df.iterrows(),
                         total=len(new_orders_df),
                         desc="Simulating assignments"):
        order_num = order['order_number']
        order_ts = pd.to_datetime(order['order_created_timestamp'])
        eval_ts = order_ts + timedelta(minutes=delta_minutes)
        r_lat, r_lon = order['restaurant_lat'], order['restaurant_lon']
        eval_hour = eval_ts.hour

        # Available couriers at eval_ts
        avail = courier_df[courier_df['courier_location_timestamp'] <= eval_ts]
        if avail.empty:
            continue

        # Build candidate list
        candidates = []
        for cid in avail['courier_id'].unique():
            locs = avail[avail['courier_id'] == cid]
            if locs.empty:
                continue
            latest = locs.iloc[-1]
            dist_km = evaluator.haversine_km(
                r_lat, r_lon, latest['courier_lat'], latest['courier_lon']
            )
            candidates.append((
                cid,
                dist_km,
                int(latest['open_orders']),
                latest['courier_lat'],
                latest['courier_lon']
            ))

        if not candidates:
            continue

        # Top-K by distance
        candidates = sorted(candidates, key=lambda x: x[1])[:3]

        # DQN selection
        best_courier, pred_cost = evaluator.select_best_courier(candidates,
                                                                eval_hour)
        if best_courier != -1:
            results.append({
                'order_number': order_num,
                'predicted_courier': best_courier,
                'predicted_cost': pred_cost
            })

    return pd.DataFrame(results)


# =============================================================================
# 6. COST REDUCTION METRICS
# =============================================================================
def compute_cost_metrics(results_df: pd.DataFrame, feat_df: pd.DataFrame) -> \
Dict[str, float]:
    if results_df.empty:
        return {'model_cost': 0.0, 'historical_cost': 0.0,
                'reduction_pct': 0.0}

    model_cost = results_df['predicted_cost'].sum()

    hist_cost = feat_df[
        feat_df['order_number'].isin(results_df['order_number'])
    ]['cost'].sum()

    reduction = (
                            hist_cost - model_cost) / hist_cost * 100 if hist_cost > 0 else 0.0

    return {
        'model_cost': model_cost,
        'historical_cost': hist_cost,
        'reduction_pct': reduction
    }


# =============================================================================
# 7. MAIN ENTRY POINT (used by interview_test_real_time.py)
# =============================================================================
def run_real_time_evaluation(
        df_filtered: pd.DataFrame,
        feat_df_new: pd.DataFrame,
        run_configuration
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Full real-time evaluation pipeline.

    Args:
        df_filtered: Raw courier tracking data
        feat_df_new: Preprocessed features with 'cost'
        run_configuration: Config namedtuple from JSON

    Returns:
        (results_df, metrics_dict)
    """
    logging.info("Starting Real-Time Evaluation (Inference Only)")

    # === 1. Load test episodes ===
    episodes_path = run_configuration.TEST_EPISODES_PATH
    if not Path(episodes_path).exists():
        raise FileNotFoundError(f"Test episodes not found: {episodes_path}")

    with open(episodes_path, 'rb') as f:
        test_episodes = pickle.load(f)
    logging.info(f"Loaded {len(test_episodes)} test episodes")

    # === 2. Load model & norm stats ===
    model_path = run_configuration.MODEL_LOAD_PATH
    checkpoint = torch.load(model_path, map_location=DEVICE)
    norm_stats = {
        'cost_mean': checkpoint['cost_mean'],
        'cost_std': checkpoint['cost_std'],
        'open_mean': checkpoint['open_mean'],
        'open_std': checkpoint['open_std']
    }
    evaluator = RealTimeEvaluator(model_path, norm_stats)

    # === 3. Prepare data ===
    courier_df = add_open_orders_tracking(df_filtered)
    new_orders_df = build_new_orders_from_episodes(test_episodes, feat_df_new)

    # === 4. Run simulation ===
    results_df = run_evaluation(
        evaluator=evaluator,
        new_orders_df=new_orders_df,
        courier_df=courier_df,
        delta_minutes=run_configuration.TIME_WINDOW_MIN
    )

    # === 5. Compute metrics ===
    metrics = compute_cost_metrics(results_df, feat_df_new)

    # === 6. Print results ===
    print("\n" + "=" * 64)
    print("REAL-TIME COURIER ASSIGNMENT SIMULATION")
    print("=" * 64)
    print(f"Orders processed:     {len(results_df):>8,}")
    print(f"Model total cost:     ${metrics['model_cost']:>10,.2f}")
    print(f"Historical cost:      ${metrics['historical_cost']:>10,.2f}")
    print(f"COST REDUCTION:        {metrics['reduction_pct']:>7.2f}%")
    print("=" * 64 + "\n")

    logging.info(
        f"Evaluation complete: {metrics['reduction_pct']:.2f}% cost reduction")

    return results_df, metrics