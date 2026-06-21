import pandas as pd
import os

class DataLoader:
    """
    Class responsible only for loading CSV files into pandas DataFrames.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found at: {self.file_path}")
        return pd.read_csv(self.file_path)
