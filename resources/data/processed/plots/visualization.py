import os
import sys
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import DateEntry

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from resources.data.process import DataLoader

class DemandDataRepository:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None

    def load_data(self) -> None:
        loader = DataLoader(self.csv_path)
        self.df = loader.load()
        self.df['date'] = pd.to_datetime(self.df['date'])

    def get_unique_articles(self) -> list[str]:
        if self.df is None:
            return []
        return sorted(self.df['article_id'].unique().tolist())

    def get_filtered_data(self, article_ids: list[str]) -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()
        return self.df[self.df['article_id'].isin(article_ids)]

class DemandPlotGenerator:
    def generate_chart(self, df: pd.DataFrame, article_ids: list[str], resolution: str, width: float, height: float) -> plt.Figure:
        fig = plt.Figure(figsize=(width, height), dpi=100)
        ax = fig.add_subplot(111)
        
        max_qty = 0
        for art_id in article_ids:
            df_art = df[df['article_id'] == art_id]
            if df_art.empty:
                continue
                
            if resolution == 'day':
                df_grouped = df_art.groupby('date')['total_quantity'].sum().reset_index()
                df_grouped = df_grouped.sort_values(by='date')
                max_qty = max(max_qty, df_grouped['total_quantity'].max())
                ax.plot(df_grouped['date'], df_grouped['total_quantity'], label=f'{art_id} (Daily)', alpha=0.8, linewidth=1.2)
            elif resolution == 'month':
                df_grouped = df_art.groupby(['year', 'month'])['total_quantity'].sum().reset_index()
                df_grouped['date'] = pd.to_datetime(df_grouped['year'].astype(str) + '-' + df_grouped['month'].astype(str) + '-01')
                df_grouped = df_grouped.sort_values(by='date')
                max_qty = max(max_qty, df_grouped['total_quantity'].max())
                ax.plot(df_grouped['date'], df_grouped['total_quantity'], label=f'{art_id} (Monthly)', marker='o', alpha=0.8, linewidth=1.8)
            elif resolution == 'year':
                df_grouped = df_art.groupby('year')['total_quantity'].sum().reset_index()
                df_grouped = df_grouped.sort_values(by='year')
                max_qty = max(max_qty, df_grouped['total_quantity'].max())
                ax.bar(df_grouped['year'].astype(str), df_grouped['total_quantity'], label=f'{art_id} (Yearly)', alpha=0.7)
                
        ax.set_title(f"Medication Demand Progression Over Time ({resolution.capitalize()})", fontsize=14, fontweight='bold')
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

    def display_chart(self, fig) -> None:
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
            
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

class Visualization:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(base_dir, "..", "training_data.csv"))
        
        self.repository = DemandDataRepository(self.data_path)
        self.repository.load_data()
        
        self.plot_generator = DemandPlotGenerator()
        
        self.root = None
        self.chart_frame = None
        self.plot_canvas = None
        self.selected_articles = []
        self.selection_vars = {}
        self.selected_df = pd.DataFrame()
        
        self.res_day = None
        self.res_month = None
        self.res_year = None
        self.start_date_entry = None
        self.end_date_entry = None
        self.scrollable_plot = None
        
        self.x_scale_value = 30.0
        self.y_scale_value = 5.2

    def start(self, article_ids: str | list[str] = None) -> None:
        self.root = tk.Tk()
        self.root.title("Medication Demand Dashboard")
        self.root.state('zoomed')
        
        nav_bar = tk.Frame(self.root, height=50, bg='#f0f2f5', padx=20)
        nav_bar.pack(side="top", fill="x")
        nav_bar.pack_propagate(False)
        
        app_title = tk.Label(nav_bar, text="Medication Demand Analytics", fg="#1a1a24", bg="#f0f2f5", font=("Arial", 11, "bold"))
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
            self.selected_df = self.repository.get_filtered_data(self.selected_articles)
            self.update_chart()
            
        self.root.mainloop()

    def handle_resolution_change(self, clicked: str) -> None:
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

    def get_date_filtered_df(self, df: pd.DataFrame) -> pd.DataFrame:
        start_str = self.start_date_entry.get().strip()
        end_str = self.end_date_entry.get().strip()
        
        filtered_df = df.copy()
        
        try:
            start_date = pd.to_datetime(start_str, format="%d/%m/%Y")
            filtered_df = filtered_df[filtered_df['date'] >= start_date]
        except Exception:
            pass
            
        try:
            end_date = pd.to_datetime(end_str, format="%d/%m/%Y")
            filtered_df = filtered_df[filtered_df['date'] <= end_date]
        except Exception:
            pass
            
        return filtered_df

    def show_settings_modal(self) -> None:
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

    def submit_settings(self, modal, x_scale: tk.Scale, y_scale: tk.Scale) -> None:
        self.x_scale_value = float(x_scale.get())
        self.y_scale_value = float(y_scale.get())
        modal.destroy()
        self.update_chart()

    def show_modal(self) -> None:
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

    def submit_selection(self, modal) -> None:
        self.selected_articles = [art_id for art_id, var in self.selection_vars.items() if var.get()]
        modal.destroy()
        self.selected_df = self.repository.get_filtered_data(self.selected_articles)
        self.update_chart()

    def update_chart(self) -> None:
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
            
        filtered_df = self.get_date_filtered_df(self.selected_df)
        
        if filtered_df.empty:
            self.placeholder_label = tk.Label(self.scrollable_plot.scrollable_frame, text="No data found within the selected date range.", font=("Arial", 14), bg='white', fg='gray')
            self.placeholder_label.pack(expand=True, pady=200, padx=200)
            return
            
        fig = self.plot_generator.generate_chart(filtered_df, self.selected_articles, resolution, width, height)
        self.scrollable_plot.display_chart(fig)

if __name__ == "__main__":
    runner = Visualization()
    runner.start()