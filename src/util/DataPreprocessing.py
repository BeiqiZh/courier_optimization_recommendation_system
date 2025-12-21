# src/util/DataPreprocessing.py
import logging
from math import radians, sin, cos, sqrt, atan2
from typing import Tuple

import pandas as pd
from collections import namedtuple


class DataPreprocessing:
    """
    End-to-end preprocessing for the courier-assignment modelling task.
    """

    def __init__(self, df: pd.DataFrame, run_configuration: namedtuple):
        self.raw_df = df.copy()
        self.config = run_configuration
        self.df_filtered: pd.DataFrame | None = None
        self.feat_df: pd.DataFrame | None = None
        self.feat_df_new: pd.DataFrame | None = None

    # --------------------------------------------------------------------- #
    # 1. Helper: Haversine distance (meters to km)
    # --------------------------------------------------------------------- #
    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2) -> float:
        R = 6371000
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    # --------------------------------------------------------------------- #
    # 2. Add lookup timestamps
    # --------------------------------------------------------------------- #
    @staticmethod
    def _add_order_lookup_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['order_lookup_timestamps'] = (
            df.groupby('order_number')['courier_location_timestamp'].shift(1)
        )
        first_mask = df.groupby('order_number').cumcount() == 0
        df.loc[first_mask, 'order_lookup_timestamps'] = df.loc[first_mask, 'order_created_timestamp']
        return df

    # --------------------------------------------------------------------- #
    # 3. Main pipeline
    # --------------------------------------------------------------------- #
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logging.info("Starting DataPreprocessing pipeline")

        df = self.raw_df.sort_values(['order_number',
                                      'courier_location_timestamp'])\
            .reset_index(drop=True)

        #ADD THIS BLOCK
        timestamp_cols = [
            'courier_location_timestamp',
            'order_created_timestamp'
        ]

        for col in timestamp_cols:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

        # --- 24-hour window ---
        start_time = pd.to_datetime('2021-04-01 06:00:00+00:00', utc=True)
        end_time = pd.to_datetime('2021-04-02 06:00:00+00:00', utc=True)

        mask = (
            df['courier_location_timestamp'].between(start_time, end_time) &
            df['order_created_timestamp'].between(start_time, end_time)
        )
        self.df_filtered = df[mask].copy()
        logging.info(f"Filtered dataframe: {len(self.df_filtered)} rows")

        # --- Order stats ---
        order_stats = (
            self.df_filtered.groupby('order_number')
            .agg({
                'courier_id': ['nunique', 'last'],
                'courier_location_timestamp': 'count'
            })
            .reset_index()
        )
        order_stats.columns = ['order_number', 'num_couriers', 'last_courier', 'timestamp_count']
        order_stats['reassignment'] = (order_stats['num_couriers'] > 1).astype(int)

        # --- Binary target ---
        last_couriers = self.df_filtered.groupby('order_number')['courier_id'].transform('last')
        self.df_filtered['order_assignment_unsuccess_binary'] = (
            self.df_filtered['courier_id'] != last_couriers
        ).astype(int)

        # --- Lookup timestamps ---
        self.df_filtered = self._add_order_lookup_timestamps(self.df_filtered)

        # --- Distance to restaurant ---
        self.df_filtered['distance_km'] = self.df_filtered.apply(
            lambda r: self._haversine_m(
                r['courier_lat'], r['courier_lon'],
                r['restaurant_lat'], r['restaurant_lon']
            ) / 1000,
            axis=1
        )

        # --- Unique (order, courier) attempts ---
        self.feat_df = self.df_filtered.drop_duplicates(
            subset=['order_number', 'courier_id']
        ).copy()

        # --- Generate candidates using config ---
        df_candidates = self._add_two_nearest_candidates(
            feat_df=self.feat_df,
            df_full=self.df_filtered,
            time_window_min=self.config.TIME_WINDOW_MIN,
            n_candidates=self.config.N_CANDIDATES
        )

        # --- Combine real + candidates ---
        real = self.feat_df[[
            "order_number", "courier_id", "courier_location_timestamp",
            "order_lookup_timestamps", "distance_km",
            "order_assignment_unsuccess_binary", "restaurant_lon",
            "order_created_timestamp", "restaurant_lat"
        ]].copy()

        self.feat_df_new = pd.concat([real, df_candidates], ignore_index=True)
        self.feat_df_new = self.feat_df_new.sort_values(
            ['order_number', 'order_lookup_timestamps']
        ).reset_index(drop=True)

        # --- Cost ---
        self.feat_df_new['cost'] = self.feat_df_new['distance_km'] * self.config.RATE_PER_KM
        self.feat_df["cost"] = self.feat_df['distance_km'] * self.config.RATE_PER_KM

        # --- Open orders ---
        self.feat_df_new['open_orders'] = self.feat_df_new.apply(
            lambda r: self._count_open_orders(
                courier_id=r['courier_id'],
                ts=r['courier_location_timestamp'],
                df=self.feat_df_new
            ),
            axis=1
        )

        # --- Hour ---
        self.feat_df_new['hour'] = self.feat_df_new['order_lookup_timestamps'].dt.hour

        logging.info(
            f"Preprocessing finished → "
            f"feat_df_new: {len(self.feat_df_new)} rows, "
            f"feat_df: {len(self.feat_df)} rows, "
            f"df_filtered: {len(self.df_filtered)} rows"
        )
        return self.feat_df_new, self.feat_df, self.df_filtered

    # --------------------------------------------------------------------- #
    # Candidate generation
    # --------------------------------------------------------------------- #
    @staticmethod
    def _add_two_nearest_candidates(
        feat_df: pd.DataFrame,
        df_full: pd.DataFrame,
        time_window_min: int,
        n_candidates: int
    ) -> pd.DataFrame:
        df_courier = df_full[[
            'courier_id', 'courier_location_timestamp', 'distance_km'
        ]].copy()

        candidate_rows = []
        # Extract order metadata once (from feat_df, which has it)
        order_metadata = feat_df.drop_duplicates('order_number')[
            ['order_number', 'order_created_timestamp', 'restaurant_lat',
             'restaurant_lon']
        ].set_index('order_number')

        for _, row in feat_df.iterrows():
            order_num = row['order_number']
            assigned_id = row['courier_id']
            t_center = row['order_lookup_timestamps']
            t_min = t_center - pd.Timedelta(minutes=time_window_min)
            t_max = t_center + pd.Timedelta(minutes=time_window_min)

            win = df_courier[
                df_courier['courier_location_timestamp'].between(t_min, t_max)
            ]
            if win.empty:
                continue

            snap = (
                win.sort_values('courier_location_timestamp')
                   .drop_duplicates('courier_id', keep='last')
            )
            snap = snap[snap['courier_id'] != assigned_id]
            if len(snap) < n_candidates:
                continue

            nearest = snap.nsmallest(n_candidates, 'distance_km')
            for _, cand in nearest.iterrows():
                candidate_rows.append({
                    'order_number'                    : order_num,
                    'courier_id'                      : cand['courier_id'],
                    'order_lookup_timestamps'         : t_center,
                    'courier_location_timestamp'     : cand['courier_location_timestamp'],
                    'distance_km'                     : cand['distance_km'],
                    'order_assignment_unsuccess_binary': -1,
                    # === Add metadata ===
                    'order_created_timestamp': order_metadata.at[
                        order_num, 'order_created_timestamp'],
                    'restaurant_lat': order_metadata.at[
                        order_num, 'restaurant_lat'],
                    'restaurant_lon': order_metadata.at[
                        order_num, 'restaurant_lon']
                })

        return pd.DataFrame(candidate_rows)

    # --------------------------------------------------------------------- #
    # Count open orders
    # --------------------------------------------------------------------- #
    @staticmethod
    def _count_open_orders(courier_id, ts, df: pd.DataFrame) -> int:
        ongoing = df[
            (df['courier_id'] == courier_id) &
            (df['order_lookup_timestamps'] <= ts) &
            (df['courier_location_timestamp'] >= ts)
            ]
        return len(ongoing)