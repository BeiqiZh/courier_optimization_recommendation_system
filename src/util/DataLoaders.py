import logging
from abc import ABC, abstractmethod
import os.path
import pandas as pd


class AbstractDataLoader(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_data(self, filename):
        logging.info('Checking file exists.')

        if not os.path.isfile(filename):
            logging.error('File does not exist')
            # TODO: raise exception
            raise FileNotFoundError(f'File {filename} does not exist')
        else:
            logging.info('Found file: ' + filename)


class FileDataLoader(AbstractDataLoader):

    # Initialization
    def __init__(self, filename: str):
        super().__init__()
        logging.info('Initializing Data Loading')
        self.filename = filename

    # Load data from file and return data
    def load_data(self):
        # TODO: Check file exists
        super().load_data(self.filename)

        # TODO: Load data from file
        logging.info('Loading data using pandas')
        try:
            df = pd.read_csv(self.filename,
                             parse_dates=['courier_location_timestamp',
                                          'order_created_timestamp'])
            logging.info(
                f'Successfully loaded {len(df)} rows from {self.filename}')
            return df  # TODO: Return your data object here
        except Exception as e:
            logging.error(f'Failed to load data: {str(e)}')
            raise
