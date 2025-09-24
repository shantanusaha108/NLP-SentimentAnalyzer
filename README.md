# Sentiment Insight Dashboard  

A **Desktop Sentiment Analysis App** that combines the simplicity of **VADER** with the deep learning power of **RoBERTa**.  
Built with **Python** + **Tkinter**, this app allows you to analyze text sentiment manually or in batch CSV mode and explore insights with interactive dashboards.

---

## Features
- Manual Sentiment Analysis – Enter text and analyze instantly.  
- Batch CSV Analysis – Upload datasets for bulk processing.  
- Interactive Dashboards – Includes:
  - Pie Charts (sentiment distribution)  
  - Histograms (score distributions)  
  - Scatter Plots (model comparison)  
  - Correlation Heatmaps (VADER vs RoBERTa)  
- Export Results – Save combined results to CSV.  

---

## Tech Stack  

| Technology | Usage |
|------------|--------|
| <img src="https://www.vectorlogo.zone/logos/python/python-icon.svg" width="40" height="40"/> | Core programming language |
| <img src="https://upload.wikimedia.org/wikipedia/commons/8/88/TkInterPython.png" width="40" height="40"/> | GUI framework |
| <img src="https://pandas.pydata.org/static/img/pandas_mark.svg" width="40" height="40"/> | Data handling |
| <img src="https://matplotlib.org/_static/logo2.svg" width="40" height="40"/> | Visualizations |
| <img src="https://seaborn.pydata.org/_images/logo-wide-lightbg.svg" width="80" height="30"/> | Advanced plots |
| <img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/NTLK_Logo.png" width="40" height="40"/> | VADER sentiment analysis |
| <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="40" height="40"/> | RoBERTa transformer model |

---

## Screenshots & Graphs  

### Main App Window  
![Main App](./screenshots/Screenshot%202025-09-23%20225234.png)  
Perform manual text analysis or upload a CSV file. Progress is shown with a loading bar.  

---

### Dashboard Overview  
![Dashboard](./screenshots/Screenshot%202025-09-24%20113206.png)  
The dashboard displays sentiment results and provides multiple graphs for better insights.  

---

### Graph Types  

#### 1. Pie Charts  
![Pie Chart](./screenshots/Screenshot%202025-09-24%20113206.png)  
- Compare **VADER vs RoBERTa** sentiment distribution.  
- Shows proportions of Positive, Negative, and Neutral sentiments.  

#### 2. Histograms  
![Histogram](./screenshots/Screenshot%202025-09-24%20113206.png)  
- Displays the **distribution of sentiment scores** (e.g., VADER compound, RoBERTa probabilities).  
- Helps identify skewness in the dataset.  

#### 3. Scatter Plots  
![Scatter Plot](./screenshots/Screenshot%202025-09-24%20113206.png)  
- Plots **VADER compound scores** against **RoBERTa positive scores**.  
- Highlights agreement or disagreement between the two models.  

#### 4. Correlation Heatmap  
![Heatmap](./screenshots/Screenshot%202025-09-24%20113303.png)  
- Shows correlations between all sentiment metrics:  
  - VADER (neg, neu, pos, compound)  
  - RoBERTa (negative, neutral, positive)  
- Useful for comparing how both models align.  

---

## Installation  

```bash
git clone https://github.com/shantanusaha108/Sentiment-Insight-Dashboard.git
cd Sentiment-Insight-Dashboard
pip install -r requirements.txt
python app.py
