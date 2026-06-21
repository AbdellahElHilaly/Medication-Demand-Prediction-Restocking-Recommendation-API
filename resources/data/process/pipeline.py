import os
import pandas as pd
from .loader import DataLoader
from .cleaner import DataCleaner
from .aggregator import DataAggregator
from .engineer import FeatureEngineer
from .data_visualizer import DataVisualizer

class DataPipeline:
    """
    Orchestration class that coordinates all data processing steps:
    1. Loading the raw CSV dataset.
    2. Aggregating data.
    3. Filling missing dates with 0.
    4. Cleaning.
    5. Feature Engineering.
    6. Saving the processed training data to CSV.
    7. Generating diagnostic plots.
    """
    def __init__(self, raw_csv_path: str, processed_csv_path: str, plots_dir: str):
        self.raw_csv_path = raw_csv_path
        self.processed_csv_path = processed_csv_path
        self.plots_dir = plots_dir

    def run(self, sample_article_id: str = "ART001") -> pd.DataFrame:
        print("[Pipeline] Starting data processing pipeline...")
        
        # 1. Load
        loader = DataLoader(self.raw_csv_path)
        df_raw = loader.load()
        print(f"[Pipeline] Loaded raw data: {df_raw.shape[0]} rows.")
        
        # 2. Clean (Converts date to datetime and ensures total_quantity is integer)
        cleaner = DataCleaner()
        df_cleaned = cleaner.clean(df_raw)
        print(f"[Pipeline] Cleaned raw data: {df_cleaned.shape[0]} rows.")
        
        # 3. Aggregate
        aggregator = DataAggregator()
        df_agg = aggregator.aggregate_daily(df_cleaned)
        print(f"[Pipeline] Aggregated daily data: {df_agg.shape[0]} rows.")
        
        # 4. Complete Date Sequence & Fill Missing Dates
        df_completed = aggregator.fill_missing_dates(df_agg)
        print(f"[Pipeline] Filled missing dates: {df_completed.shape[0]} rows.")
        
        # 5. Feature Engineering
        engineer = FeatureEngineer()
        df_features = engineer.add_features(df_completed)
        print(f"[Pipeline] Added features. Columns: {list(df_features.columns)}")
        
        # 6. Save final processed dataset
        os.makedirs(os.path.dirname(self.processed_csv_path), exist_ok=True)
        df_features.to_csv(self.processed_csv_path, index=False)
        print(f"[Pipeline] Saved training data to: {self.processed_csv_path}")
        
        # 7. Visualize
        visualizer = DataVisualizer(self.plots_dir)
        try:
            # Single article visualization
            art_plot = visualizer.plot_article_demand(df_features, sample_article_id)
            print(f"[Pipeline] Generated demand plot for {sample_article_id} at: {art_plot}")
            
            # Multi-article visualization test
            sample_list = ["ART001", "ART002", "ART003"]
            multi_plot = visualizer.plot_article_demand(df_features, sample_list)
            print(f"[Pipeline] Generated multi-article demand plot for {sample_list} at: {multi_plot}")
            
            dist_plot = visualizer.plot_monthly_distribution(df_features)
            print(f"[Pipeline] Generated monthly distribution plot at: {dist_plot}")
        except Exception as e:
            print(f"[Pipeline] Visualization warning: {e}")
            
        print("[Pipeline] Data pipeline completed successfully!")
        return df_features
