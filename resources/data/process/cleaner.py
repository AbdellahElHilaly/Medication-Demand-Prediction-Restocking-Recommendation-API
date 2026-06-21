import pandas as pd

class DataCleaner:
    """
    Class responsible only for cleaning the raw dataframe:
    - Standardizing column types (converting date to datetime).
    - Checking and handling missing values (filling with 0 or dropping).
    - Sorting by article_id and date.
    """
    def __init__(self, date_col: str = "date", quantity_col: str = "total_quantity", article_col: str = "article_id"):
        self.date_col = date_col
        self.quantity_col = quantity_col
        self.article_col = article_col

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        
        # Convert date column to datetime
        df_cleaned[self.date_col] = pd.to_datetime(df_cleaned[self.date_col])
        
        # Convert quantity column to numeric (integer)
        df_cleaned[self.quantity_col] = pd.to_numeric(df_cleaned[self.quantity_col], errors='coerce').fillna(0).astype(int)
        
        # Handle missing article IDs by dropping them
        df_cleaned = df_cleaned.dropna(subset=[self.article_col])
        
        # Sort by article_id and date
        df_cleaned = df_cleaned.sort_values(by=[self.article_col, self.date_col]).reset_index(drop=True)
        
        return df_cleaned
