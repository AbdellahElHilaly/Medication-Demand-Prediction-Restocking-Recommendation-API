import pandas as pd

class DataAggregator:
    """
    Class responsible only for aggregating demand data.
    E.g. Grouping by article_id and date to ensure there are no duplicates.
    Can also be extended to create continuous daily time series for each article,
    filling missing dates with 0.
    """
    def __init__(self, date_col: str = "date", article_col: str = "article_id", quantity_col: str = "total_quantity"):
        self.date_col = date_col
        self.article_col = article_col
        self.quantity_col = quantity_col

    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        df_agg = df.groupby([self.article_col, self.date_col]).agg({
            self.quantity_col: 'sum'
        }).reset_index()
        return df_agg

    def fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures a complete daily sequence for each article, filling missing dates with 0 quantity.
        """
        articles = df[self.article_col].unique()
        min_date = df[self.date_col].min()
        max_date = df[self.date_col].max()
        
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        multi_idx = pd.MultiIndex.from_product([articles, full_date_range], names=[self.article_col, self.date_col])
        
        df_reindexed = df.set_index([self.article_col, self.date_col]).reindex(multi_idx).reset_index()
        df_reindexed[self.quantity_col] = df_reindexed[self.quantity_col].fillna(0).astype(int)
        
        return df_reindexed
