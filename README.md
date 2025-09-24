# 🧠 Sentiment Insight Dashboard  

A **Desktop Sentiment Analysis App** that combines the simplicity of **VADER** with the deep learning power of **RoBERTa**.  
Built with **Python** + **Tkinter**, this app allows you to analyze text sentiment **manually** or in **batch CSV mode** and explore insights with **interactive dashboards**.

---

## 🚀 Features
- ✍️ **Manual Sentiment Analysis** – Enter text and analyze instantly.  
- 📂 **Batch CSV Analysis** – Upload datasets for bulk processing.  
- 📊 **Interactive Dashboards** – Includes:
  - Pie Charts (sentiment distribution)  
  - Histograms (score distributions)  
  - Scatter Plots (model comparison)  
  - Correlation Heatmaps (VADER vs RoBERTa)  
- 💾 **Export Results** – Save combined results to CSV.  

---

## 🛠️ Tech Stack  

| Technology | Usage |
|------------|--------|
| ![Python](https://www.vectorlogo.zone/logos/python/python-icon.svg) | Core programming language |
| ![Tkinter](https://upload.wikimedia.org/wikipedia/commons/8/88/TkInterPython.png) | GUI framework |
| ![Pandas](https://pandas.pydata.org/static/img/pandas_mark.svg) | Data handling |
| ![Matplotlib](https://matplotlib.org/_static/logo2.svg) | Visualizations |
| ![Seaborn](https://seaborn.pydata.org/_images/logo-wide-lightbg.svg) | Advanced plots |
| ![NLTK](https://upload.wikimedia.org/wikipedia/commons/3/3a/NTLK_Logo.png) | VADER sentiment analysis |
| ![HuggingFace](https://huggingface.co/front/assets/huggingface_logo-noborder.svg) | RoBERTa transformer model |

---

## 📷 Screenshots  

### 🖼️ Main App Window  
![App Screenshot](./screenshots/Screenshot%202025-09-23%20225234.png)  
Perform **manual text analysis** or upload a **CSV file**. Progress is shown with a loading bar.  

---

### 📊 Sentiment Dashboard  
![Dashboard Screenshot](./screenshots/Screenshot%202025-09-24%20113206.png)  
- **Top Table**: Displays individual text results (VADER & RoBERTa).  
- **Pie Charts**: Sentiment distribution comparison between VADER & RoBERTa.  

---

### 🔥 Correlation Heatmap  
![Heatmap Screenshot](./screenshots/Screenshot%202025-09-24%20113303.png)  
- Shows correlation between **VADER (neg, neu, pos, compound)** and **RoBERTa scores**.  
- Helps understand how the models align or differ.  

---

## ⚡ Installation  

```bash
git clone https://github.com/your-username/Sentiment-Insight-Dashboard.git
cd Sentiment-Insight-Dashboard
pip install -r requirements.txt
python app.py
