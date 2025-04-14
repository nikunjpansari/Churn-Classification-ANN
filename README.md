# Churn-Classification-Improved-ANN

A robust web application that leverages an enhanced Artificial Neural Network (ANN) model to predict customer churn. This project builds on previous architectures with improvements in model performance and interpretability, allowing businesses to identify at-risk customers and make informed retention decisions.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Churn-Classification-Improved-ANN is designed to provide an accurate, end-to-end solution for predicting customer churn. Built using state-of-the-art deep learning techniques with TensorFlow and Keras, this application offers clear insights into the reasons behind churn by providing detailed predictions and metrics. Whether you’re a data scientist or a business analyst, this tool helps monitor customer retention and reduce revenue loss.

## Features

- **Enhanced Predictive Model:** Utilizes an improved ANN architecture with tuned hyperparameters for higher accuracy in churn prediction.
- **Real-Time Prediction:** Input customer data and receive immediate churn risk classification.
- **Detailed Metrics:** 
  - **Prediction Confidence:** Overall certainty of the churn prediction.
  - **Probability Breakdown:** Class probabilities for the predicted churn and non-churn scenarios.
  - **Feature Importance:** Insights into which factors contribute most to customer churn.
- **User-Friendly Interface:** Clean and responsive UI built using Streamlit for an interactive experience.
- **Data Preprocessing Pipeline:** Built-in preprocessing steps to handle missing values and perform normalization.

## Installation

### Prerequisites

- Python 3.9+
- pip

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/nikunjpansari/Churn-Classification-ANN.git
   cd Churn-Classification-ANN

2. **Set Up a Virtual Environment**

      ```bash
   python -m venv venv
   source venv/bin/activate

3. **Install Required Packages:**

      ```bash
   pip install -r requirements.txt

### **Usage**

   ```bash
    streamlit run app.py
