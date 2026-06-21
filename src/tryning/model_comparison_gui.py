import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import DateEntry

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.enging.medication_demand_predictor import MedicationDemandPredictor

class DemandDataRepository:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.df['date'] = pd.to_datetime(self.df['date'])

    def get_unique_articles(self):
        if self.df is None:
            return []
        return sorted(self.df['article_id'].unique().tolist())

    def get_filtered_data(self, article_ids):
        if self.df is None:
            return pd.DataFrame()
        return self.df[self.df['article_id'].isin(article_ids)]

class ModelDemandPlotGenerator:
    def __init__(self, model_path, scalers_path):
        self.model_path = model_path
        self.scalers_path = scalers_path
        self.model = MedicationDemandPredictor(
            input_size=4,
            hidden_size=64,
            num_layers=2,
            forecast_horizon=7
        )
        self.reload_model_and_scalers()

    def reload_model_and_scalers(self):
        with open(self.scalers_path, "rb") as file:
            self.scalers = pickle.load(file)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def generate_predictions(self, df_art):
        feature_columns = ["total_quantity", "month", "dayofweek", "is_weekend"]
        predicted_quantities = [np.nan] * len(df_art)
        
        for i in range(14, len(df_art)):
            sequence_df = df_art.iloc[i - 14 : i].copy()
            
            sequence_data = []
            for col in feature_columns:
                val = sequence_df[[col]].copy()
                val_scaled = self.scalers[col].transform(val)
                sequence_data.append(val_scaled)
                
            sequence_data = np.hstack(sequence_data)
            input_tensor = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                pred_scaled = self.model(input_tensor).squeeze(0).numpy()
                
            pred_real = self.scalers["total_quantity"].inverse_transform(pred_scaled[0].reshape(-1, 1))[0, 0]
            pred_real = max(0.0, pred_real)
            predicted_quantities[i] = pred_real
            
        df_art["predicted_quantity"] = predicted_quantities
        return df_art

    def generate_chart(self, predicted_dfs, article_ids, start_date, end_date, resolution, width, height):
        fig = plt.Figure(figsize=(width, height), dpi=100)
        ax = fig.add_subplot(111)
        
        max_qty = 0
        
        for idx, art_id in enumerate(article_ids):
            if art_id not in predicted_dfs:
                continue
            df_art = predicted_dfs[art_id].copy()
            
            if start_date:
                df_art = df_art[df_art['date'] >= start_date]
            if end_date:
                df_art = df_art[df_art['date'] <= end_date]
                
            if df_art.empty:
                continue
                
            real_color = plt.cm.tab20((idx * 2 + 1) % 20)
            pred_color = plt.cm.tab20((idx * 2) % 20)
            
            if resolution == 'day':
                df_grouped = df_art.groupby('date')[['total_quantity', 'predicted_quantity']].sum().reset_index()
                df_grouped = df_grouped.sort_values(by='date')
                max_qty = max(max_qty, df_grouped['total_quantity'].max(), df_grouped['predicted_quantity'].max())
                
                ax.plot(df_grouped['date'], df_grouped['total_quantity'], label=f'{art_id} Real', alpha=0.3, linewidth=1.0, color=real_color)
                ax.plot(df_grouped['date'], df_grouped['predicted_quantity'], label=f'{art_id} Pred', alpha=1.0, linewidth=2.0, color=pred_color)
                
            elif resolution == 'month':
                df_grouped = df_art.groupby(['year', 'month'])[['total_quantity', 'predicted_quantity']].sum().reset_index()
                df_grouped['date'] = pd.to_datetime(df_grouped['year'].astype(str) + '-' + df_grouped['month'].astype(str) + '-01')
                df_grouped = df_grouped.sort_values(by='date')
                max_qty = max(max_qty, df_grouped['total_quantity'].max(), df_grouped['predicted_quantity'].max())
                
                ax.plot(df_grouped['date'], df_grouped['total_quantity'], label=f'{art_id} Real', marker='o', alpha=0.3, linewidth=1.2, color=real_color)
                ax.plot(df_grouped['date'], df_grouped['predicted_quantity'], label=f'{art_id} Pred', marker='x', alpha=1.0, linewidth=2.5, color=pred_color)
                
            elif resolution == 'year':
                df_grouped = df_art.groupby('year')[['total_quantity', 'predicted_quantity']].sum().reset_index()
                df_grouped = df_grouped.sort_values(by='year')
                max_qty = max(max_qty, df_grouped['total_quantity'].max(), df_grouped['predicted_quantity'].max())
                
                x = np.arange(len(df_grouped))
                bar_width = 0.35
                
                ax.bar(x - bar_width/2, df_grouped['total_quantity'], bar_width, label=f'{art_id} Real', alpha=0.25, color=real_color)
                ax.bar(x + bar_width/2, df_grouped['predicted_quantity'], bar_width, label=f'{art_id} Pred', alpha=1.0, color=pred_color)
                ax.set_xticks(x)
                ax.set_xticklabels(df_grouped['year'].astype(str))
                
        ax.set_title(f"Medication Demand Progression vs AI Model Forecast ({resolution.capitalize()})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Quantity Consumed", fontsize=12)
        
        if max_qty > 0:
            ax.set_ylim(0, max_qty * 1.15)
            
        ax.legend(fontsize=10, loc="upper right")
        fig.tight_layout()
        return fig

class ScrollablePlotFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg='white')
        self.scrollbar_x = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='white')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set)
        
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="top", fill="both", expand=True)
        
        self.plot_canvas = None

    def display_chart(self, fig):
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
            
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

class ModelComparisonVisualization:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(base_dir, "..", "..", "resources", "data", "processed", "training_data.csv"))
        self.model_path = os.path.abspath(os.path.join(base_dir, "..", "..", "resources", "models", "medication_demand_lstm.pth"))
        self.scalers_path = os.path.abspath(os.path.join(base_dir, "..", "..", "resources", "models", "scalers.pkl"))
        
        self.repository = DemandDataRepository(self.data_path)
        self.repository.load_data()
        
        self.plot_generator = ModelDemandPlotGenerator(self.model_path, self.scalers_path)
        
        self.root = None
        self.chart_frame = None
        self.plot_canvas = None
        self.selected_articles = []
        self.selection_vars = {}
        self.predicted_dfs = {}
        
        self.res_day = None
        self.res_month = None
        self.res_year = None
        self.start_date_entry = None
        self.end_date_entry = None
        self.scrollable_plot = None
        
        self.x_scale_value = 30.0
        self.y_scale_value = 5.2

    def start(self, article_ids = None):
        self.root = tk.Tk()
        self.root.title("Medication Demand & AI Regression Comparator")
        self.root.state('zoomed')
        
        nav_bar = tk.Frame(self.root, height=50, bg='#f0f2f5', padx=20)
        nav_bar.pack(side="top", fill="x")
        nav_bar.pack_propagate(False)
        
        app_title = tk.Label(nav_bar, text="Medication Demand Analytics & AI Forecast", fg="#1a1a24", bg="#f0f2f5", font=("Arial", 11, "bold"))
        app_title.pack(side="left")
        
        select_button = tk.Button(nav_bar, text="Select Medication Articles", command=self.show_modal, bg="royalblue", fg="white", font=("Arial", 10, "bold"), padx=12, pady=4, relief="flat")
        select_button.pack(side="right", padx=(10, 0))
        
        settings_button = tk.Button(nav_bar, text="⚙️ Parameters", command=self.show_settings_modal, bg="#e9ecef", fg="#1a1a24", font=("Arial", 10, "bold"), padx=10, pady=4, relief="flat")
        settings_button.pack(side="right", padx=10)
        
        self.res_day = tk.BooleanVar(value=True)
        self.res_month = tk.BooleanVar(value=False)
        self.res_year = tk.BooleanVar(value=False)
        
        res_frame = tk.Frame(nav_bar, bg="#f0f2f5")
        res_frame.pack(side="left", padx=15)
        
        cb_day = tk.Checkbutton(res_frame, text="Day", variable=self.res_day, command=lambda: self.handle_resolution_change('day'), bg="#f0f2f5", fg="#1a1a24", font=("Arial", 9, "bold"), activebackground="#f0f2f5", activeforeground="#1a1a24")
        cb_day.pack(side="left", padx=3)
        
        cb_month = tk.Checkbutton(res_frame, text="Month", variable=self.res_month, command=lambda: self.handle_resolution_change('month'), bg="#f0f2f5", fg="#1a1a24", font=("Arial", 9, "bold"), activebackground="#f0f2f5", activeforeground="#1a1a24")
        cb_month.pack(side="left", padx=3)
        
        cb_year = tk.Checkbutton(res_frame, text="Year", variable=self.res_year, command=lambda: self.handle_resolution_change('year'), bg="#f0f2f5", fg="#1a1a24", font=("Arial", 9, "bold"), activebackground="#f0f2f5", activeforeground="#1a1a24")
        cb_year.pack(side="left", padx=3)
        
        filter_frame = tk.Frame(nav_bar, bg="#f0f2f5")
        filter_frame.pack(side="right", padx=10)
        
        tk.Label(filter_frame, text="From:", bg="#f0f2f5", fg="#1a1a24", font=("Arial", 9, "bold")).pack(side="left", padx=2)
        
        min_date = self.repository.df['date'].min()
        max_date = self.repository.df['date'].max()
        
        self.start_date_entry = DateEntry(filter_frame, width=10, background='royalblue', foreground='white', borderwidth=2, date_pattern='dd/mm/yyyy', mindate=min_date, maxdate=max_date)
        self.start_date_entry.set_date(min_date)
        self.start_date_entry.pack(side="left", padx=2)
        
        tk.Label(filter_frame, text="To:", bg="#f0f2f5", fg="#1a1a24", font=("Arial", 9, "bold")).pack(side="left", padx=2)
        
        self.end_date_entry = DateEntry(filter_frame, width=10, background='royalblue', foreground='white', borderwidth=2, date_pattern='dd/mm/yyyy', mindate=min_date, maxdate=max_date)
        self.end_date_entry.set_date(max_date)
        self.end_date_entry.pack(side="left", padx=2)
        
        apply_btn = tk.Button(filter_frame, text="Filter", command=self.update_chart, bg="#28a745", fg="white", font=("Arial", 9, "bold"), padx=8, relief="flat")
        apply_btn.pack(side="left", padx=5)
        
        self.start_date_entry.bind("<Return>", lambda e: self.update_chart())
        self.end_date_entry.bind("<Return>", lambda e: self.update_chart())
        
        self.chart_frame = tk.Frame(self.root, bg='white')
        self.chart_frame.pack(side="bottom", fill="both", expand=True)
        
        self.scrollable_plot = ScrollablePlotFrame(self.chart_frame)
        self.scrollable_plot.pack(fill="both", expand=True)
        
        self.placeholder_label = tk.Label(self.scrollable_plot.scrollable_frame, text="No articles selected. Click 'Select Medication Articles' to load data.", font=("Arial", 14), bg='white', fg='gray')
        self.placeholder_label.pack(expand=True, pady=200, padx=200)
        
        if article_ids:
            if isinstance(article_ids, str):
                self.selected_articles = [article_ids]
            else:
                self.selected_articles = list(article_ids)
            self.calculate_predictions()
            self.update_chart()
            
        self.root.mainloop()

    def handle_resolution_change(self, clicked):
        if clicked == 'day':
            if not self.res_day.get():
                self.res_day.set(True)
            self.res_month.set(False)
            self.res_year.set(False)
        elif clicked == 'month':
            if not self.res_month.get():
                self.res_month.set(True)
            self.res_day.set(False)
            self.res_year.set(False)
        elif clicked == 'year':
            if not self.res_year.get():
                self.res_year.set(True)
            self.res_day.set(False)
            self.res_month.set(False)
        self.update_chart()

    def get_dates(self):
        start_str = self.start_date_entry.get().strip()
        end_str = self.end_date_entry.get().strip()
        start_date = None
        end_date = None
        try:
            start_date = pd.to_datetime(start_str, format="%d/%m/%Y")
        except:
            pass
        try:
            end_date = pd.to_datetime(end_str, format="%d/%m/%Y")
        except:
            pass
        return start_date, end_date

    def show_settings_modal(self):
        modal = tk.Toplevel(self.root)
        modal.title("Scale Parameters")
        modal.geometry("320x240")
        modal.configure(bg="#f8f9fa")
        modal.transient(self.root)
        modal.grab_set()
        
        header = tk.Label(modal, text="⚙️ Scale Parameters", fg="#1a1a24", bg="#f8f9fa", font=("Arial", 11, "bold"), pady=10)
        header.pack(side="top")
        
        x_scale = tk.Scale(modal, from_=10, to_=60, orient="horizontal", label="X-Scale (Width)", bg="#f8f9fa", fg="#1a1a24", font=("Arial", 9, "bold"), highlightthickness=0, resolution=1)
        x_scale.set(int(self.x_scale_value))
        x_scale.pack(fill="x", padx=20, pady=5)
        
        y_scale = tk.Scale(modal, from_=3, to_=10, orient="horizontal", label="Y-Scale (Height)", bg="#f8f9fa", fg="#1a1a24", font=("Arial", 9, "bold"), highlightthickness=0, resolution=0.1)
        y_scale.set(self.y_scale_value)
        y_scale.pack(fill="x", padx=20, pady=5)
        
        submit_btn = tk.Button(modal, text="Submit", command=lambda: self.submit_settings(modal, x_scale, y_scale), bg="royalblue", fg="white", font=("Arial", 10, "bold"), padx=15, pady=5, relief="flat")
        submit_btn.pack(side="bottom", pady=15)

    def submit_settings(self, modal, x_scale, y_scale):
        self.x_scale_value = float(x_scale.get())
        self.y_scale_value = float(y_scale.get())
        modal.destroy()
        self.update_chart()

    def show_modal(self):
        modal = tk.Toplevel(self.root)
        modal.title("Select Medication Articles")
        modal.geometry("750x550")
        modal.configure(bg="#f8f9fa")
        modal.transient(self.root)
        modal.grab_set()
        
        header = tk.Label(modal, text="Medication Grid Selector", fg="#1a1a24", bg="#f8f9fa", font=("Arial", 12, "bold"), pady=10)
        header.pack(side="top")
        
        footer = tk.Frame(modal, bg="#f8f9fa", pady=10)
        footer.pack(side="bottom", fill="x")
        
        submit_button = tk.Button(
            footer, 
            text="Submit Selection", 
            command=lambda: self.submit_selection(modal), 
            bg="royalblue", 
            fg="white", 
            font=("Arial", 11, "bold"), 
            padx=20, 
            pady=6, 
            relief="flat"
        )
        submit_button.pack(pady=5)
        
        grid_container = tk.Frame(modal, bg="#f8f9fa", padx=15, pady=5)
        grid_container.pack(fill="both", expand=True)
        
        unique_articles = self.repository.get_unique_articles()
        
        cols = 8
        for i, art_id in enumerate(unique_articles):
            if art_id not in self.selection_vars:
                is_selected = art_id in self.selected_articles
                self.selection_vars[art_id] = tk.BooleanVar(value=is_selected)
                
            row = i // cols
            col = i % cols
            
            cb = tk.Checkbutton(
                grid_container, 
                text=art_id, 
                variable=self.selection_vars[art_id], 
                indicatoron=False, 
                relief="flat", 
                selectcolor="royalblue", 
                bg="#e9ecef", 
                fg="#1a1a24", 
                activebackground="royalblue", 
                activeforeground="white", 
                padx=4, 
                pady=3,
                font=("Arial", 8, "bold")
            )
            cb.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            
        for col_idx in range(cols):
            grid_container.grid_columnconfigure(col_idx, weight=1)

    def calculate_predictions(self):
        self.plot_generator.reload_model_and_scalers()
        self.predicted_dfs = {}
        for art_id in self.selected_articles:
            df_art = self.repository.df[self.repository.df['article_id'] == art_id].sort_values(by='date').copy()
            if not df_art.empty:
                self.predicted_dfs[art_id] = self.plot_generator.generate_predictions(df_art)

    def submit_selection(self, modal):
        self.selected_articles = [art_id for art_id, var in self.selection_vars.items() if var.get()]
        modal.destroy()
        
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
        if hasattr(self, 'placeholder_label') and self.placeholder_label:
            self.placeholder_label.destroy()
            
        self.placeholder_label = tk.Label(
            self.scrollable_plot.scrollable_frame, 
            text="⏳ Loading AI predictions & rendering chart... Please wait...", 
            font=("Arial", 14, "bold"), 
            bg='white', 
            fg='royalblue'
        )
        self.placeholder_label.pack(expand=True, pady=200, padx=200)
        self.root.update()
        
        self.calculate_predictions()
        
        if self.placeholder_label:
            self.placeholder_label.destroy()
            self.placeholder_label = None
            
        self.update_chart()

    def update_chart(self):
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
            
        if hasattr(self, 'placeholder_label') and self.placeholder_label:
            self.placeholder_label.destroy()
            self.placeholder_label = None
            
        if not self.selected_articles:
            self.placeholder_label = tk.Label(self.scrollable_plot.scrollable_frame, text="No articles selected. Click 'Select Medication Articles' to load data.", font=("Arial", 14), bg='white', fg='gray')
            self.placeholder_label.pack(expand=True, pady=200, padx=200)
            return
            
        resolution = 'day'
        if self.res_month.get():
            resolution = 'month'
        elif self.res_year.get():
            resolution = 'year'
            
        width = self.x_scale_value
        height = self.y_scale_value
            
        start_date, end_date = self.get_dates()
        
        fig = self.plot_generator.generate_chart(self.predicted_dfs, self.selected_articles, start_date, end_date, resolution, width, height)
        
        self.scrollable_plot.display_chart(fig)

if __name__ == "__main__":
    runner = ModelComparisonVisualization()
    runner.start()
