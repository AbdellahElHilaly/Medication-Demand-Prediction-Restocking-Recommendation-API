import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataVisualizer:
    """
    Class responsible only for visualizing the data and saving plots.
    """
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        sns.set_theme(style="darkgrid")

    def plot_article_demand(self, df: pd.DataFrame, article_ids: str | list[str], date_col: str = "date", quantity_col: str = "total_quantity", article_col: str = "article_id") -> str:
        """
        Plots the demand time series progression for a single article ID or a list of article IDs.
        """
        if isinstance(article_ids, str):
            ids_to_plot = [article_ids]
        else:
            ids_to_plot = list(article_ids)
            
        plt.figure(figsize=(14, 7))
        
        plotted_any = False
        for art_id in ids_to_plot:
            df_article = df[df[article_col] == art_id].sort_values(by=date_col)
            if df_article.empty:
                print(f"[Visualizer] Warning: No data found for article: {art_id}")
                continue
                
            plt.plot(df_article[date_col], df_article[quantity_col], label=f'Demand {art_id}', alpha=0.8, linewidth=1.5)
            plotted_any = True
            
        if not plotted_any:
            plt.close()
            raise ValueError(f"No data found for any of the articles: {article_ids}")
            
        title_suffix = ", ".join(ids_to_plot[:5]) + ("..." if len(ids_to_plot) > 5 else "")
        plt.title(f"Medication Demand Progression for: {title_suffix}", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Quantity Consumed", fontsize=12)
        plt.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        
        if len(ids_to_plot) == 1:
            filename = f"{ids_to_plot[0]}_demand.png"
        else:
            filename = f"multi_demand_{len(ids_to_plot)}_articles.png"
            
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    def plot_monthly_distribution(self, df: pd.DataFrame, quantity_col: str = "total_quantity") -> str:
        """
        Plots a boxplot of demand distribution by month to show seasonality.
        """
        if 'month' not in df.columns:
            raise ValueError("Dataframe must contain 'month' column. Run FeatureEngineer first.")
            
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='month', y=quantity_col, palette="crest")
        plt.title("Medication Demand Distribution by Month", fontsize=14, fontweight='bold')
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Quantity Consumed", fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, "monthly_demand_distribution.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path
