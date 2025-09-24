# Sentiment Insight Dashboard 📝📊

A **Desktop Sentiment Analysis App** that combines the simplicity of **VADER** with the deep learning power of **RoBERTa**.  
Built with **Python** + **Tkinter**, this app allows you to analyze text sentiment manually or in batch CSV mode and explore insights with interactive dashboards.

---

## Features ✨
- **Manual Sentiment Analysis** – Enter text and analyze instantly.  
- **Batch CSV Analysis** – Upload datasets for bulk processing.  
- **Interactive Dashboards** – Includes:
  - 📊 Pie Charts (sentiment distribution)  
  - 📈 Histograms (score distributions)  
  - 🔹 Scatter Plots (model comparison)  
  - 🔥 Correlation Heatmaps (VADER vs RoBERTa)  
- **Export Results** – Save combined results to CSV. 💾  

---

## Tech Stack 🛠️


| Technology | Usage |
|------------|-------|
| <img src="https://github.com/shantanusaha108/VADER-RoBERTa-SentimentApp/blob/main/screenshots/python.png" width="70" height="70"/> | **Python** – Core programming language |
| <img src="https://github.com/shantanusaha108/VADER-RoBERTa-SentimentApp/blob/main/screenshots/tkinter.png" width="70" height="70"/> | **Tkinter** – GUI framework |
| <img src="https://github.com/shantanusaha108/VADER-RoBERTa-SentimentApp/blob/main/screenshots/pandas.png" width="70" height="70"/> | **Pandas** – Data handling |
| <img src="https://github.com/shantanusaha108/VADER-RoBERTa-SentimentApp/blob/main/screenshots/matplotlib.png" width="70" height="70"/> | **Matplotlib** – Visualizations |
| <img src="https://github.com/shantanusaha108/VADER-RoBERTa-SentimentApp/blob/main/screenshots/seaborn.png" width="110" height="50"/> | **Seaborn** – Advanced plots |
| <img src="https://github.com/shantanusaha108/VADER-RoBERTa-SentimentApp/blob/main/screenshots/NLTK(Natural%20Language%20Toolkit).png" width="100" height="100"/> <br> <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="100" height="100"/> | **NLTK , Hugging Face** – VADER + RoBERTa sentiment analysis |

---

## Screenshots & Graphs 🖼️

### Main App Window 🏠
![Main App](https://github.com/shantanusaha108/VADER-RoBERTa-SentimentApp/blob/main/screenshots/Screenshot%202025-09-23%20225234.png)  
Perform manual text analysis or upload a CSV file. Progress is shown with a loading bar.  

---

### Dashboard Overview 📊
![Dashboard](./screenshots/Screenshot%202025-09-24%20113206.png)  
The dashboard displays sentiment results and provides multiple graphs for better insights.  

---

### Graph Types 📈

#### 1. Pie Charts 🥧
![Pie Chart](./screenshots/Screenshot%202025-09-24%20113206.png)  
- Compare **VADER vs RoBERTa** sentiment distribution.  
- Shows proportions of Positive, Negative, and Neutral sentiments.  

#### 2. Histograms 📊
![Histogram](https://github.com/shantanusaha108/VADER-RoBERTa-SentimentApp/blob/main/screenshots/Screenshot%202025-09-24%20113220.png)  
- Displays the **distribution of sentiment scores** (e.g., VADER compound, RoBERTa probabilities).  
- Helps identify skewness in the dataset.  

#### 3. Scatter Plots 🔹
![Scatter Plot](https://github.com/shantanusaha108/VADER-RoBERTa-SentimentApp/blob/main/screenshots/Screenshot%202025-09-24%20113234.png)  
- Plots **VADER compound scores** against **RoBERTa positive scores**.  
- Highlights agreement or disagreement between the two models.  

#### 4. Correlation Heatmap 🔥
![Heatmap](https://github.com/shantanusaha108/VADER-RoBERTa-SentimentApp/blob/main/screenshots/Screenshot%202025-09-24%20113303.png)  
- Shows correlations between all sentiment metrics:  
  - VADER (neg, neu, pos, compound)  
  - RoBERTa (negative, neutral, positive)  
- Useful for comparing how both models align.  

---

## Installation 💻

```bash
# Clone the repository
git clone https://github.com/shantanusaha108/VADER-RoBERTa-SentimentApp.git
cd VADER-RoBERTa-SentimentApp

# Install dependencies
pip install -r requirements.txt

# Run the app
python SentimentalAnalysis.py

