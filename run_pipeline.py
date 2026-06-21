import os
import sys

# Add resources to python path so it can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'resources'))

from data.process import DataPipeline

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    raw_csv = os.path.join(base_dir, 'resources', 'data', 'raw', 'Hospital Medication Demand.csv')
    processed_csv = os.path.join(base_dir, 'resources', 'data', 'processed', 'training_data.csv')
    plots_dir = os.path.join(base_dir, 'resources', 'data', 'processed', 'plots')
    
    pipeline = DataPipeline(
        raw_csv_path=raw_csv,
        processed_csv_path=processed_csv,
        plots_dir=plots_dir
    )
    
    df = pipeline.run(sample_article_id="ART001")
    
    print("\n--- Processed Dataset Preview ---")
    print(df.head())
