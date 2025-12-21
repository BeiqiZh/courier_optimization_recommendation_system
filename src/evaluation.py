"""
Real-time evaluation ONLY (no training).
"""

import logging
import json
from collections import namedtuple
import os
import sys

# -------------------------------------------------
# Fix imports so util.* works when running directly
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util.DataLoaders import FileDataLoader
from util.DataPreprocessing import DataPreprocessing
from util.RealTimeEvaluation import run_real_time_evaluation



# -------------------------------------------------
# Path helper
# -------------------------------------------------
def project_path(*paths):
    return os.path.join(PROJECT_ROOT, *paths)


# -------------------------------------------------
# Config loader
# -------------------------------------------------
def load_config():
    config_path = project_path('resources', 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        data = json.load(f)
        return namedtuple('Config', data.keys())(*data.values())


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == '__main__':

    # === Setup logging ===
    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    logging.info("Starting Real-Time Evaluation (No Training)")

    # === Load configuration ===
    try:
        run_configuration = load_config()
        logging.info(
            f"Config loaded. Model path: {run_configuration.MODEL_LOAD_PATH}"
        )
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise

    # === Load raw data ===
    data_path = project_path('data', 'courier.csv')

    try:
        data_loader = FileDataLoader(data_path)
        df_raw = data_loader.load_data()
        logging.info(f"Data loaded: {len(df_raw)} rows")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

    # === Preprocess data ===
    try:
        preprocessor = DataPreprocessing(df_raw, run_configuration)
        feat_df_new, feat_df, df_filtered = preprocessor.run()
        logging.info("Preprocessing complete")
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise

    # === Real-Time Evaluation (NO training) ===
    try:
        logging.info("Starting real-time evaluation...")

        results_df, metrics = run_real_time_evaluation(
            df_filtered=df_filtered,
            feat_df_new=feat_df_new,
            run_configuration=run_configuration
        )

        reduction_pct = metrics.get('reduction_pct', 0.0)
        logging.info(
            f"Real-time evaluation complete. Cost reduction: {reduction_pct:.2f}%"
        )

        # Save results
        results_path = project_path(
            'real_time_evaluation_results.csv'
        )
        results_df.to_csv(results_path, index=False)
        logging.info(f"Results saved to {results_path}")

    except Exception as e:
        logging.error(f'Failed to evaluate real-time model: {e}')
        raise