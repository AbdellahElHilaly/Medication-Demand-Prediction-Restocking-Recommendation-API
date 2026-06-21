import pandas as pd

class FeatureEngineer:
    """
    Class responsible only for creating features from datetime and other columns.
    Features: month, day of week, day of year, year, is_weekend, etc.
    """
    def __init__(self, date_col: str = "date"):
        self.date_col = date_col

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_featured = df.copy()
        
        # Extract features from datetime
        df_featured['year'] = df_featured[self.date_col].dt.year
        df_featured['month'] = df_featured[self.date_col].dt.month
        df_featured['day'] = df_featured[self.date_col].dt.day
        df_featured['dayofweek'] = df_featured[self.date_col].dt.dayofweek
        df_featured['dayofyear'] = df_featured[self.date_col].dt.dayofyear
        df_featured['is_weekend'] = df_featured[self.date_col].dt.dayofweek.isin([5, 6]).astype(int)
        
        return df_featured
