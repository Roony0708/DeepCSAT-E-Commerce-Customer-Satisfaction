# 📊 Customer Satisfaction Prediction

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Status](https://img.shields.io/badge/status-active-success.svg)]

## 🚀 Overview
The **Customer Satisfaction Prediction App** is a **Streamlit-based web application** that predicts Customer Satisfaction (CSAT) scores using a **pre-trained deep learning model**. This project helps businesses understand customer feedback and improve service quality by leveraging **machine learning and data analytics**.

## 🎯 Key Features
- 📊 **Predicts Customer Satisfaction Score** based on multiple customer interaction parameters.
- 🔍 **Uses a Pre-trained Deep Learning Model (TensorFlow/Keras).**
- 🛠 **Handles Various Input Features** like Channel Name, Product Category, Handling Time, Issue Type, etc.
- 📉 **Preprocesses and Normalizes Data** for better predictions.
- 📈 **Provides Real-Time Insights** with an interactive Streamlit interface.

## 🏗️ Project Structure
```
📦 Customer-Satisfaction-Prediction
 ┣ 📂 Data Sets        # Pre-trained ML models (Keras/TensorFlow)
 ┣ 📂 Document         # Sample datasets
 ┣ 📂 Notebook         # Data preprocessing scripts
 ┣ 📂 Streamlit        # Data preprocessing scripts
 ┣ 📜 README.md        # Documentation
```

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Customer-Satisfaction-Prediction.git
cd Customer-Satisfaction-Prediction
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Application
```bash
streamlit run app.py
```

## 🎯 How It Works
1. The user selects or inputs key customer details such as **Channel Name, Product Category, Handling Time, Issue Type, and more.**
2. The app **preprocesses and normalizes** the input data.
3. A **trained neural network model (Keras/TensorFlow)** predicts the **Customer Satisfaction Score.**
4. The result is displayed as a **CSAT Score (1-5 scale).**

## 📌 Technologies Used
- **Python** 🐍
- **Streamlit** 🎨 (for UI)
- **TensorFlow/Keras** 🧠 (for machine learning model)
- **Pandas & NumPy** 📊 (for data processing)
- **Joblib** 💾 (for loading pre-trained scalers and features)

## 🤖 Model Details
The application uses a **deep learning-based model trained on historical customer service interactions.** The model:
✅ Extracts important features from structured data.
✅ Uses **LSTM and Dense layers** for predictions.
✅ Predicts **CSAT Scores (1 to 5) based on past interactions.**

## 📂 Data Processing Steps
1. **Feature Engineering** – Transform categorical and numerical data.
2. **Data Normalization** – Scale numerical fields using a **pre-trained scaler (joblib).**
3. **Model Prediction** – Convert user inputs into a feature matrix and predict satisfaction score.

## 🚀 Business Impact
✅ **Improves Customer Experience** – Helps businesses enhance support quality.  
✅ **Identifies Pain Points** – Pinpoints areas needing service improvement.  
✅ **Data-Driven Decisions** – Enables organizations to act on real insights.  


## 📧 Contact
For inquiries, reach out at **ydabhi1999@gmail.com**.

