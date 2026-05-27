# 🎯 Predictive Marketing & Retention Engine

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Lifetimes-orange.svg)]()

An end-to-end Machine Learning web application designed to transform raw transactional sales data into a targeted, predictive marketing strategy. 

## 🚀 Live Demo
**[Click here to view the live application](insert-your-streamlit-url-here)** *(Make sure to replace this with your actual Streamlit link!)*

## 🧠 Overview
This tool transitions a company from looking at historical analytics to executing a predictive strategy. It automatically isolates high-value customers who have a high probability of churning and aggregates their forecasted lost revenue, allowing businesses to stop revenue leaks before they happen.

**Key Capabilities:**
* **Automated RFM Segmentation:** Ingests raw sales records and mathematically engineers Recency, Frequency, and Monetary features to cluster the user base into distinct behavioral tiers (e.g., Champions, Loyal, At-Risk).
* **Predictive CLV & Churn:** Deploys advanced statistical probability models (**BG/NBD** and **Gamma-Gamma**) to forecast the future behavior of every individual customer. It calculates their exact probability of churn and their predicted Customer Lifetime Value (CLV) over the next 90 days.
* **Actionable Export:** Marketing teams can filter this data in real-time and export highly targeted "win-back" lists for immediate campaign execution.

## 🛠️ Tech Stack
* **Language:** Python
* **Web Framework:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** `lifetimes` (BG/NBD, Gamma-Gamma Fitter)
* **Data Visualization:** Plotly Express

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/predictive-retention-engine.git](https://github.com/YourUsername/predictive-retention-engine.git)
   cd predictive-retention-engine
   Install the required dependencies:

Bash
pip install -r requirements.txt

Run the Streamlit application:

Bash
streamlit run app.py

Project Structure
app.py: The main Streamlit application script containing the front-end UI and the back-end machine learning pipeline.

online_retail.csv: The transactional dataset used to train the predictive models.

requirements.txt: The list of Python dependencies required to run the application.
Dataset
This project utilizes an Online Retail dataset containing raw transactional records. The AI pipeline cleans this data, drops invalid rows, calculates aggregate spend per invoice, and transforms the raw timeline into a mathematical matrix suitable for predictive modeling.
