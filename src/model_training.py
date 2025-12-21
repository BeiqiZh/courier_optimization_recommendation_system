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
from util.Predictors import DQNCourierPredictor

# -------------------------------------------------
# Path helpers
# -------------------------------------------------
def project_path(*paths):
    """Build absolute paths relative to project root."""
    return os.path.join(PROJECT_ROOT, *paths)


# -------------------------------------------------
# Config loader
# -------------------------------------------------
def load_config():
    config_path = project_path('resources', 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with open(config_path) as json_file:
        return json.loads(
            json_file.read(),
            object_hook=lambda d: namedtuple('Config', d.keys())(*d.values())
        )


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == '__main__':

    # Logging
    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    logging.info('Starting classification program')

    # ------------------- LOAD CONFIG -------------------
    try:
        run_configuration = load_config()
        logging.info('Configuration loaded successfully')
    except Exception as e:
        logging.error(f'Failed to load config: {e}')
        raise

    # ------------------- LOAD DATA -------------------
    data_path = project_path('data', 'courier.csv')

    try:
        data_loader = FileDataLoader(data_path)
        data = data_loader.load_data()
        logging.info('Data loaded successfully')
    except Exception as e:
        logging.error(f'Failed to load data: {e}')
        raise

    # ------------------- PREPROCESSING -------------------
    try:
        preprocessor = DataPreprocessing(data, run_configuration)
        feat_df_new, feat_df, df_filtered = preprocessor.run()
        logging.info('Data preprocessed successfully')
    except Exception as e:
        logging.error(f'Failed to preprocess data: {e}')
        raise

    # ------------------- TRAIN MODEL -------------------
    try:
        predictor = DQNCourierPredictor(
            feat_df_new=feat_df_new,
            feat_df=feat_df,
            df_filtered=df_filtered,
            run_configuration=run_configuration
        )
        predictor.train()
        logging.info('Model trained successfully')
    except Exception as e:
        logging.error(f'Failed to train model: {e}')
        raise

    logging.info('Program completed successfully')
