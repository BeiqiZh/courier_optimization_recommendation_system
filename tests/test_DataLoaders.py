# test_DataLoaders.py
import unittest
import os
import pandas as pd
from src.util.DataLoaders import FileDataLoader

class TestFileDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_file = 'test_data.csv'
        # Create a small test CSV
        pd.DataFrame({
            'courier_id': [1],
            'order_number': ['A'],
            'courier_location_timestamp': ['2025-10-23 10:00:00'],
            'courier_lat': [40.0],
            'courier_lon': [-74.0],
            'order_created_timestamp': ['2025-10-23 09:55:00'],
            'restaurant_lat': [40.1],
            'restaurant_lon': [-74.1]
        }).to_csv(self.test_file, index=False)

    def test_load_data_success(self):
        loader = FileDataLoader(self.test_file)
        data = loader.load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 1)

    def test_load_data_file_not_found(self):
        loader = FileDataLoader('nonexistent.csv')
        with self.assertRaises(FileNotFoundError):
            loader.load_data()

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

if __name__ == '__main__':
    unittest.main()