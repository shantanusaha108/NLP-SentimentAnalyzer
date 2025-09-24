<<<<<<< HEAD
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from threading import Thread
import nltk

# --- Global Configurations ---
# Suppress NLTK download message if already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon', quiet=True)

# Import and initialize VADER analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Import and initialize RoBERTa analyzer (This part can be slow on first run)
try:
    from transformers import pipeline, AutoTokenizer
    roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    roberta_analyzer = pipeline("sentiment-analysis", model=roberta_model_name)
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
    roberta_available = True
except ImportError:
    roberta_available = False
    roberta_analyzer = None
    roberta_tokenizer = None
    # Provide a warning if RoBERTa dependencies are not met
    print("Warning: Hugging Face 'transformers' library or dependencies not found. RoBERTa model will be unavailable.")

# --- Sentiment Analysis Core Functions ---

def interpret_vader(score):
    """Interprets VADER's compound score to return a sentiment label."""
    compound_score = score['compound']
    if compound_score >= 0.05:
        return "POSITIVE"
    elif compound_score <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def analyze_with_roberta(text):
    """Analyzes text with the RoBERTa model and returns a label and score, truncating if necessary."""
    if not roberta_available:
        return "Error", 0.0
    
    # Truncate the text to the model's maximum sequence length to prevent the tensor size error
    max_length = roberta_tokenizer.model_max_length
    if len(text) > max_length:
        text = text[:max_length]
        
    result = roberta_analyzer(text)[0]
    # Map RoBERTa's labels (LABEL_0, LABEL_1, LABEL_2) to standard sentiment labels
    label_map = {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE'}
    sentiment = label_map.get(result['label'], result['label'])
    return sentiment, result['score']


# --- Dashboard Window Class ---

class SentimentDashboard(tk.Toplevel):
    def __init__(self, parent, results_df):
        super().__init__(parent)
        self.title("Sentiment Analysis Dashboard")
        self.geometry("1400x900")
        self.results_df = results_df

        self.create_scrollable_frame()

    def create_scrollable_frame(self):
        # Create a Canvas and a Scrollbar
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        # Create a Frame inside the Canvas to hold all widgets
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add mouse wheel scrolling functionality
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units"))
        canvas.bind_all("<Button-4>", lambda event: canvas.yview_scroll(-1, "units")) # For Linux
        canvas.bind_all("<Button-5>", lambda event: canvas.yview_scroll(1, "units"))  # For Linux

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.create_widgets(self.scrollable_frame)

    def create_widgets(self, parent_frame):
        # --- Top: Individual Results Table ---
        table_frame = ttk.LabelFrame(parent_frame, text="Individual Analysis Results", padding="10")
        table_frame.pack(fill=tk.X, expand=True, padx=20, pady=10)
        
        columns = ("Text", "VADER Sentiment", "VADER Score", "RoBERTa Sentiment", "RoBERTa Score")
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor=tk.CENTER)
        self.tree.column("Text", width=600, anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.populate_table()
        
        # --- Middle: Visualizations ---
        # Pie Charts
        pie_charts_frame = ttk.Frame(parent_frame)
        pie_charts_frame.pack(fill=tk.X, expand=True, padx=20, pady=10)
        self.create_pie_charts(pie_charts_frame)

        # Histograms
        histograms_frame = ttk.Frame(parent_frame)
        histograms_frame.pack(fill=tk.X, expand=True, padx=20, pady=10)
        self.create_histograms(histograms_frame)
        
        # Scatter Plots
        scatter_plots_frame = ttk.Frame(parent_frame)
        scatter_plots_frame.pack(fill=tk.X, expand=True, padx=20, pady=10)
        self.create_scatter_plots(scatter_plots_frame)
        
        # Heatmap
        heatmap_frame = ttk.Frame(parent_frame)
        heatmap_frame.pack(fill=tk.X, expand=True, padx=20, pady=10)
        self.create_heatmap(heatmap_frame)

        # Download Button
        ttk.Button(parent_frame, text="Download Combined Results CSV", command=self.download_csv).pack(pady=20)

    def populate_table(self):
        for index, row in self.results_df.iterrows():
            self.tree.insert("", "end", values=(
                row['Text'],
                row['VADER_Sentiment'],
                f"{row['VADER_compound']:.4f}",
                row['RoBERTa_Sentiment'],
                f"{row['RoBERTa_Score']:.4f}"
            ))

    def create_pie_charts(self, parent_frame):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # VADER Pie Chart
        vader_counts = self.results_df['VADER_Sentiment'].value_counts()
        vader_labels = vader_counts.index
        vader_sizes = vader_counts.values
        axes[0].pie(vader_sizes, labels=vader_labels, autopct='%1.1f%%', startangle=90)
        axes[0].axis('equal')
        axes[0].set_title("VADER Sentiment Distribution")

        # RoBERTa Pie Chart
        roberta_counts = self.results_df['RoBERTa_Sentiment'].value_counts()
        roberta_labels = roberta_counts.index
        roberta_sizes = roberta_counts.values
        axes[1].pie(roberta_sizes, labels=roberta_labels, autopct='%1.1f%%', startangle=90)
        axes[1].axis('equal')
        axes[1].set_title("RoBERTa Sentiment Distribution")
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, parent_frame)
        toolbar.update()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

    def create_histograms(self, parent_frame):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # VADER Histograms
        sns.histplot(self.results_df['VADER_neg'], ax=axes[0, 0], kde=True, color='red', label='VADER')
        axes[0, 0].set_title('VADER Negative Scores')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        sns.histplot(self.results_df['VADER_neu'], ax=axes[0, 1], kde=True, color='gray', label='VADER')
        axes[0, 1].set_title('VADER Neutral Scores')
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        sns.histplot(self.results_df['VADER_pos'], ax=axes[0, 2], kde=True, color='green', label='VADER')
        axes[0, 2].set_title('VADER Positive Scores')
        axes[0, 2].grid(True)
        axes[0, 2].legend()

        # RoBERTa Histograms (using confidence scores per sentiment)
        sns.histplot(self.results_df[self.results_df['RoBERTa_Sentiment'] == 'NEGATIVE']['RoBERTa_Score'], ax=axes[1, 0], kde=True, color='red', label='RoBERTa')
        axes[1, 0].set_title('RoBERTa Neg Confidence')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        sns.histplot(self.results_df[self.results_df['RoBERTa_Sentiment'] == 'NEUTRAL']['RoBERTa_Score'], ax=axes[1, 1], kde=True, color='gray', label='RoBERTa')
        axes[1, 1].set_title('RoBERTa Neu Confidence')
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        sns.histplot(self.results_df[self.results_df['RoBERTa_Sentiment'] == 'POSITIVE']['RoBERTa_Score'], ax=axes[1, 2], kde=True, color='green', label='RoBERTa')
        axes[1, 2].set_title('RoBERTa Pos Confidence')
        axes[1, 2].grid(True)
        axes[1, 2].legend()
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent_frame)
        toolbar.update()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

    def create_scatter_plots(self, parent_frame):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Prepare data for plotting by melting the DataFrame into a long format
        pos_df = self.results_df[['VADER_pos', 'RoBERTa_Score']].copy()
        pos_df['VADER_pos_Sentiment'] = self.results_df['VADER_Sentiment']
        pos_df['RoBERTa_pos_Sentiment'] = self.results_df['RoBERTa_Sentiment']
        
        neg_df = self.results_df[['VADER_neg', 'RoBERTa_Score']].copy()
        neg_df['VADER_neg_Sentiment'] = self.results_df['VADER_Sentiment']
        neg_df['RoBERTa_neg_Sentiment'] = self.results_df['RoBERTa_Sentiment']
        
        neu_df = self.results_df[['VADER_neu', 'RoBERTa_Score']].copy()
        neu_df['VADER_neu_Sentiment'] = self.results_df['VADER_Sentiment']
        neu_df['RoBERTa_neu_Sentiment'] = self.results_df['RoBERTa_Sentiment']

        # Filter data for each sentiment type
        pos_vader_data = pos_df[pos_df['VADER_pos_Sentiment'] == 'POSITIVE']
        pos_roberta_data = pos_df[pos_df['RoBERTa_pos_Sentiment'] == 'POSITIVE']

        neg_vader_data = neg_df[neg_df['VADER_neg_Sentiment'] == 'NEGATIVE']
        neg_roberta_data = neg_df[neg_df['RoBERTa_neg_Sentiment'] == 'NEGATIVE']
        
        neu_vader_data = neu_df[neu_df['VADER_neu_Sentiment'] == 'NEUTRAL']
        neu_roberta_data = neu_df[neu_df['RoBERTa_neu_Sentiment'] == 'NEUTRAL']

        # Positive Scores Scatter Plot
        axes[0].scatter(pos_vader_data.index, pos_vader_data['VADER_pos'], c='green', alpha=0.5, label='VADER')
        axes[0].scatter(pos_roberta_data.index, pos_roberta_data['RoBERTa_Score'], c='lime', alpha=0.5, marker='x', label='RoBERTa')
        axes[0].set_title("Positive Scores Comparison")
        axes[0].set_xlabel("Data Point Index")
        axes[0].set_ylabel("Score")
        axes[0].grid(True)
        axes[0].legend()

        # Negative Scores Scatter Plot
        axes[1].scatter(neg_vader_data.index, neg_vader_data['VADER_neg'], c='red', alpha=0.5, label='VADER')
        axes[1].scatter(neg_roberta_data.index, neg_roberta_data['RoBERTa_Score'], c='salmon', alpha=0.5, marker='x', label='RoBERTa')
        axes[1].set_title("Negative Scores Comparison")
        axes[1].set_xlabel("Data Point Index")
        axes[1].set_ylabel("Score")
        axes[1].grid(True)
        axes[1].legend()

        # Neutral Scores Scatter Plot
        axes[2].scatter(neu_vader_data.index, neu_vader_data['VADER_neu'], c='blue', alpha=0.5, label='VADER')
        axes[2].scatter(neu_roberta_data.index, neu_roberta_data['RoBERTa_Score'], c='cyan', alpha=0.5, marker='x', label='RoBERTa')
        axes[2].set_title("Neutral Scores Comparison")
        axes[2].set_xlabel("Data Point Index")
        axes[2].set_ylabel("Score")
        axes[2].grid(True)
        axes[2].legend()
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent_frame)
        toolbar.update()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
    
    def create_heatmap(self, parent_frame):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Ensure columns are numeric and handle any non-numeric values
        numeric_cols = ['VADER_neg', 'VADER_neu', 'VADER_pos', 'VADER_compound', 'RoBERTa_Score']
        df_for_corr = self.results_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        correlation_matrix = df_for_corr.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap of All Scores")

        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent_frame)
        toolbar.update()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

    def download_csv(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            try:
                self.results_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results successfully saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"An error occurred while saving the file: {e}")


# --- Main Application Class ---

class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Desktop Sentiment Analysis App")
        self.geometry("850x250")
        
        self.results_df = pd.DataFrame()
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Removed model selection buttons, as both models are now used for a single, combined analysis
        ttk.Label(main_frame, text="Analysis will be performed using both VADER and RoBERTa models.", font=("Helvetica", 10, "italic")).pack(pady=(0, 10))

        manual_frame = ttk.LabelFrame(main_frame, text="Manual Text Analysis", padding="10")
        manual_frame.pack(fill=tk.X, pady=(0, 10))
        self.text_input = tk.Text(manual_frame, height=4, width=60)
        self.text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        ttk.Button(manual_frame, text="Analyze Text", command=self.analyze_manual_text).pack(side=tk.RIGHT)

        batch_frame = ttk.LabelFrame(main_frame, text="Batch CSV Analysis", padding="10")
        batch_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(batch_frame, text="Upload CSV", command=self.upload_csv).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 0))
        self.progress_bar = ttk.Progressbar(batch_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 0))
        self.status_label = ttk.Label(batch_frame, text="Ready.")
        self.status_label.pack(side=tk.LEFT, padx=5)

    def update_status(self, message):
        self.status_label.config(text=message)
        self.update_idletasks()

    def analyze_text(self, text):
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_sentiment = interpret_vader(vader_scores)
        roberta_sentiment, roberta_score = analyze_with_roberta(text)

        return {
            'Text': text,
            'VADER_Sentiment': vader_sentiment,
            'VADER_compound': vader_scores['compound'],
            'VADER_pos': vader_scores['pos'],
            'VADER_neg': vader_scores['neg'],
            'VADER_neu': vader_scores['neu'],
            'RoBERTa_Sentiment': roberta_sentiment,
            'RoBERTa_Score': roberta_score
        }

    def analyze_manual_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Input Error", "Please enter some text to analyze.")
            return

        results_dict = self.analyze_text(text)
        self.results_df = pd.DataFrame([results_dict])
        
        if not self.results_df.empty:
            SentimentDashboard(self, self.results_df)
        self.update_status("Manual analysis complete.")

    def analyze_batch(self, file_path):
        self.progress_bar.stop()
        self.progress_bar['value'] = 0
        self.status_label.config(text="Processing...")
        self.update_idletasks()

        try:
            df = pd.read_csv(file_path, header=None, names=['id', 'text', 'extra_col'])
            
            if 'text' not in df.columns:
                messagebox.showerror("File Error", "The CSV file must contain a 'text' column.")
                self.update_status("Error: 'text' column not found.")
                return

            results = []
            total_rows = len(df)

            self.progress_bar['maximum'] = total_rows
            self.progress_bar.start()

            for index, row in df.iterrows():
                text = str(row['text'])
                results_dict = self.analyze_text(text)
                results.append(results_dict)

                self.progress_bar['value'] = index + 1
                self.update_idletasks()

            self.results_df = pd.DataFrame(results)
            
            self.progress_bar.stop()
            self.update_status("Batch analysis complete.")
            
            if not self.results_df.empty:
                SentimentDashboard(self, self.results_df)

        except Exception as e:
            self.progress_bar.stop()
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
            self.update_status("Error during batch analysis.")

    def upload_csv(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            self.update_status("Loading and analyzing CSV...")
            Thread(target=self.analyze_batch, args=(file_path,)).start()

if __name__ == "__main__":
    app = SentimentApp()
=======
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from threading import Thread
import nltk

# --- Global Configurations ---
# Suppress NLTK download message if already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon', quiet=True)

# Import and initialize VADER analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Import and initialize RoBERTa analyzer (This part can be slow on first run)
try:
    from transformers import pipeline, AutoTokenizer
    roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    roberta_analyzer = pipeline("sentiment-analysis", model=roberta_model_name)
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
    roberta_available = True
except ImportError:
    roberta_available = False
    roberta_analyzer = None
    roberta_tokenizer = None
    # Provide a warning if RoBERTa dependencies are not met
    print("Warning: Hugging Face 'transformers' library or dependencies not found. RoBERTa model will be unavailable.")

# --- Sentiment Analysis Core Functions ---

def interpret_vader(score):
    """Interprets VADER's compound score to return a sentiment label."""
    compound_score = score['compound']
    if compound_score >= 0.05:
        return "POSITIVE"
    elif compound_score <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def analyze_with_roberta(text):
    """Analyzes text with the RoBERTa model and returns a label and score, truncating if necessary."""
    if not roberta_available:
        return "Error", 0.0
    
    # Truncate the text to the model's maximum sequence length to prevent the tensor size error
    max_length = roberta_tokenizer.model_max_length
    if len(text) > max_length:
        text = text[:max_length]
        
    result = roberta_analyzer(text)[0]
    # Map RoBERTa's labels (LABEL_0, LABEL_1, LABEL_2) to standard sentiment labels
    label_map = {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE'}
    sentiment = label_map.get(result['label'], result['label'])
    return sentiment, result['score']


# --- Dashboard Window Class ---

class SentimentDashboard(tk.Toplevel):
    def __init__(self, parent, results_df):
        super().__init__(parent)
        self.title("Sentiment Analysis Dashboard")
        self.geometry("1400x900")
        self.results_df = results_df

        self.create_scrollable_frame()

    def create_scrollable_frame(self):
        # Create a Canvas and a Scrollbar
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        # Create a Frame inside the Canvas to hold all widgets
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add mouse wheel scrolling functionality
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units"))
        canvas.bind_all("<Button-4>", lambda event: canvas.yview_scroll(-1, "units")) # For Linux
        canvas.bind_all("<Button-5>", lambda event: canvas.yview_scroll(1, "units"))  # For Linux

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.create_widgets(self.scrollable_frame)

    def create_widgets(self, parent_frame):
        # --- Top: Individual Results Table ---
        table_frame = ttk.LabelFrame(parent_frame, text="Individual Analysis Results", padding="10")
        table_frame.pack(fill=tk.X, expand=True, padx=20, pady=10)
        
        columns = ("Text", "VADER Sentiment", "VADER Score", "RoBERTa Sentiment", "RoBERTa Score")
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor=tk.CENTER)
        self.tree.column("Text", width=600, anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.populate_table()
        
        # --- Middle: Visualizations ---
        # Pie Charts
        pie_charts_frame = ttk.Frame(parent_frame)
        pie_charts_frame.pack(fill=tk.X, expand=True, padx=20, pady=10)
        self.create_pie_charts(pie_charts_frame)

        # Histograms
        histograms_frame = ttk.Frame(parent_frame)
        histograms_frame.pack(fill=tk.X, expand=True, padx=20, pady=10)
        self.create_histograms(histograms_frame)
        
        # Scatter Plots
        scatter_plots_frame = ttk.Frame(parent_frame)
        scatter_plots_frame.pack(fill=tk.X, expand=True, padx=20, pady=10)
        self.create_scatter_plots(scatter_plots_frame)
        
        # Heatmap
        heatmap_frame = ttk.Frame(parent_frame)
        heatmap_frame.pack(fill=tk.X, expand=True, padx=20, pady=10)
        self.create_heatmap(heatmap_frame)

        # Download Button
        ttk.Button(parent_frame, text="Download Combined Results CSV", command=self.download_csv).pack(pady=20)

    def populate_table(self):
        for index, row in self.results_df.iterrows():
            self.tree.insert("", "end", values=(
                row['Text'],
                row['VADER_Sentiment'],
                f"{row['VADER_compound']:.4f}",
                row['RoBERTa_Sentiment'],
                f"{row['RoBERTa_Score']:.4f}"
            ))

    def create_pie_charts(self, parent_frame):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # VADER Pie Chart
        vader_counts = self.results_df['VADER_Sentiment'].value_counts()
        vader_labels = vader_counts.index
        vader_sizes = vader_counts.values
        axes[0].pie(vader_sizes, labels=vader_labels, autopct='%1.1f%%', startangle=90)
        axes[0].axis('equal')
        axes[0].set_title("VADER Sentiment Distribution")

        # RoBERTa Pie Chart
        roberta_counts = self.results_df['RoBERTa_Sentiment'].value_counts()
        roberta_labels = roberta_counts.index
        roberta_sizes = roberta_counts.values
        axes[1].pie(roberta_sizes, labels=roberta_labels, autopct='%1.1f%%', startangle=90)
        axes[1].axis('equal')
        axes[1].set_title("RoBERTa Sentiment Distribution")
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, parent_frame)
        toolbar.update()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

    def create_histograms(self, parent_frame):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # VADER Histograms
        sns.histplot(self.results_df['VADER_neg'], ax=axes[0, 0], kde=True, color='red', label='VADER')
        axes[0, 0].set_title('VADER Negative Scores')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        sns.histplot(self.results_df['VADER_neu'], ax=axes[0, 1], kde=True, color='gray', label='VADER')
        axes[0, 1].set_title('VADER Neutral Scores')
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        sns.histplot(self.results_df['VADER_pos'], ax=axes[0, 2], kde=True, color='green', label='VADER')
        axes[0, 2].set_title('VADER Positive Scores')
        axes[0, 2].grid(True)
        axes[0, 2].legend()

        # RoBERTa Histograms (using confidence scores per sentiment)
        sns.histplot(self.results_df[self.results_df['RoBERTa_Sentiment'] == 'NEGATIVE']['RoBERTa_Score'], ax=axes[1, 0], kde=True, color='red', label='RoBERTa')
        axes[1, 0].set_title('RoBERTa Neg Confidence')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        sns.histplot(self.results_df[self.results_df['RoBERTa_Sentiment'] == 'NEUTRAL']['RoBERTa_Score'], ax=axes[1, 1], kde=True, color='gray', label='RoBERTa')
        axes[1, 1].set_title('RoBERTa Neu Confidence')
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        sns.histplot(self.results_df[self.results_df['RoBERTa_Sentiment'] == 'POSITIVE']['RoBERTa_Score'], ax=axes[1, 2], kde=True, color='green', label='RoBERTa')
        axes[1, 2].set_title('RoBERTa Pos Confidence')
        axes[1, 2].grid(True)
        axes[1, 2].legend()
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent_frame)
        toolbar.update()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

    def create_scatter_plots(self, parent_frame):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Prepare data for plotting by melting the DataFrame into a long format
        pos_df = self.results_df[['VADER_pos', 'RoBERTa_Score']].copy()
        pos_df['VADER_pos_Sentiment'] = self.results_df['VADER_Sentiment']
        pos_df['RoBERTa_pos_Sentiment'] = self.results_df['RoBERTa_Sentiment']
        
        neg_df = self.results_df[['VADER_neg', 'RoBERTa_Score']].copy()
        neg_df['VADER_neg_Sentiment'] = self.results_df['VADER_Sentiment']
        neg_df['RoBERTa_neg_Sentiment'] = self.results_df['RoBERTa_Sentiment']
        
        neu_df = self.results_df[['VADER_neu', 'RoBERTa_Score']].copy()
        neu_df['VADER_neu_Sentiment'] = self.results_df['VADER_Sentiment']
        neu_df['RoBERTa_neu_Sentiment'] = self.results_df['RoBERTa_Sentiment']

        # Filter data for each sentiment type
        pos_vader_data = pos_df[pos_df['VADER_pos_Sentiment'] == 'POSITIVE']
        pos_roberta_data = pos_df[pos_df['RoBERTa_pos_Sentiment'] == 'POSITIVE']

        neg_vader_data = neg_df[neg_df['VADER_neg_Sentiment'] == 'NEGATIVE']
        neg_roberta_data = neg_df[neg_df['RoBERTa_neg_Sentiment'] == 'NEGATIVE']
        
        neu_vader_data = neu_df[neu_df['VADER_neu_Sentiment'] == 'NEUTRAL']
        neu_roberta_data = neu_df[neu_df['RoBERTa_neu_Sentiment'] == 'NEUTRAL']

        # Positive Scores Scatter Plot
        axes[0].scatter(pos_vader_data.index, pos_vader_data['VADER_pos'], c='green', alpha=0.5, label='VADER')
        axes[0].scatter(pos_roberta_data.index, pos_roberta_data['RoBERTa_Score'], c='lime', alpha=0.5, marker='x', label='RoBERTa')
        axes[0].set_title("Positive Scores Comparison")
        axes[0].set_xlabel("Data Point Index")
        axes[0].set_ylabel("Score")
        axes[0].grid(True)
        axes[0].legend()

        # Negative Scores Scatter Plot
        axes[1].scatter(neg_vader_data.index, neg_vader_data['VADER_neg'], c='red', alpha=0.5, label='VADER')
        axes[1].scatter(neg_roberta_data.index, neg_roberta_data['RoBERTa_Score'], c='salmon', alpha=0.5, marker='x', label='RoBERTa')
        axes[1].set_title("Negative Scores Comparison")
        axes[1].set_xlabel("Data Point Index")
        axes[1].set_ylabel("Score")
        axes[1].grid(True)
        axes[1].legend()

        # Neutral Scores Scatter Plot
        axes[2].scatter(neu_vader_data.index, neu_vader_data['VADER_neu'], c='blue', alpha=0.5, label='VADER')
        axes[2].scatter(neu_roberta_data.index, neu_roberta_data['RoBERTa_Score'], c='cyan', alpha=0.5, marker='x', label='RoBERTa')
        axes[2].set_title("Neutral Scores Comparison")
        axes[2].set_xlabel("Data Point Index")
        axes[2].set_ylabel("Score")
        axes[2].grid(True)
        axes[2].legend()
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent_frame)
        toolbar.update()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
    
    def create_heatmap(self, parent_frame):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Ensure columns are numeric and handle any non-numeric values
        numeric_cols = ['VADER_neg', 'VADER_neu', 'VADER_pos', 'VADER_compound', 'RoBERTa_Score']
        df_for_corr = self.results_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        correlation_matrix = df_for_corr.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap of All Scores")

        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent_frame)
        toolbar.update()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

    def download_csv(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            try:
                self.results_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results successfully saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"An error occurred while saving the file: {e}")


# --- Main Application Class ---

class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Desktop Sentiment Analysis App")
        self.geometry("850x250")
        
        self.results_df = pd.DataFrame()
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Removed model selection buttons, as both models are now used for a single, combined analysis
        ttk.Label(main_frame, text="Analysis will be performed using both VADER and RoBERTa models.", font=("Helvetica", 10, "italic")).pack(pady=(0, 10))

        manual_frame = ttk.LabelFrame(main_frame, text="Manual Text Analysis", padding="10")
        manual_frame.pack(fill=tk.X, pady=(0, 10))
        self.text_input = tk.Text(manual_frame, height=4, width=60)
        self.text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        ttk.Button(manual_frame, text="Analyze Text", command=self.analyze_manual_text).pack(side=tk.RIGHT)

        batch_frame = ttk.LabelFrame(main_frame, text="Batch CSV Analysis", padding="10")
        batch_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(batch_frame, text="Upload CSV", command=self.upload_csv).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 0))
        self.progress_bar = ttk.Progressbar(batch_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 0))
        self.status_label = ttk.Label(batch_frame, text="Ready.")
        self.status_label.pack(side=tk.LEFT, padx=5)

    def update_status(self, message):
        self.status_label.config(text=message)
        self.update_idletasks()

    def analyze_text(self, text):
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_sentiment = interpret_vader(vader_scores)
        roberta_sentiment, roberta_score = analyze_with_roberta(text)

        return {
            'Text': text,
            'VADER_Sentiment': vader_sentiment,
            'VADER_compound': vader_scores['compound'],
            'VADER_pos': vader_scores['pos'],
            'VADER_neg': vader_scores['neg'],
            'VADER_neu': vader_scores['neu'],
            'RoBERTa_Sentiment': roberta_sentiment,
            'RoBERTa_Score': roberta_score
        }

    def analyze_manual_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Input Error", "Please enter some text to analyze.")
            return

        results_dict = self.analyze_text(text)
        self.results_df = pd.DataFrame([results_dict])
        
        if not self.results_df.empty:
            SentimentDashboard(self, self.results_df)
        self.update_status("Manual analysis complete.")

    def analyze_batch(self, file_path):
        self.progress_bar.stop()
        self.progress_bar['value'] = 0
        self.status_label.config(text="Processing...")
        self.update_idletasks()

        try:
            df = pd.read_csv(file_path, header=None, names=['id', 'text', 'extra_col'])
            
            if 'text' not in df.columns:
                messagebox.showerror("File Error", "The CSV file must contain a 'text' column.")
                self.update_status("Error: 'text' column not found.")
                return

            results = []
            total_rows = len(df)

            self.progress_bar['maximum'] = total_rows
            self.progress_bar.start()

            for index, row in df.iterrows():
                text = str(row['text'])
                results_dict = self.analyze_text(text)
                results.append(results_dict)

                self.progress_bar['value'] = index + 1
                self.update_idletasks()

            self.results_df = pd.DataFrame(results)
            
            self.progress_bar.stop()
            self.update_status("Batch analysis complete.")
            
            if not self.results_df.empty:
                SentimentDashboard(self, self.results_df)

        except Exception as e:
            self.progress_bar.stop()
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
            self.update_status("Error during batch analysis.")

    def upload_csv(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            self.update_status("Loading and analyzing CSV...")
            Thread(target=self.analyze_batch, args=(file_path,)).start()

if __name__ == "__main__":
    app = SentimentApp()
>>>>>>> 330afa470b74457963298fce6b80c27c4e1c0479
    app.mainloop()