# Housing Prices Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZY2STt_2uzzvB24JrNRzJaAMkET4O7x_?usp=sharing)

This project predicts housing prices using various regression techniques including data preprocessing, feature selection, and model comparison. It provides a comprehensive walkthrough of the machine learning pipeline with detailed explanations and visualizations.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation and Setup](#installation-and-setup)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Results](#results)

## Project Overview
The goal is to build predictive models that estimate the price of homes using multiple regression techniques. The project follows a structured approach from exploratory data analysis, preprocessing, feature engineering, to model evaluation and comparison.

## Installation and Setup

### 1. Prerequisites
Python 3.7+ and Jupyter environment (VS Code with Jupyter extension, JupyterLab, or Google Colab)

### 2. Dependency Installation
Install all required libraries using pip:

```bash
pip install pandas numpy seaborn scikit-learn matplotlib kagglehub statsmodels
```

### 3. Running the Project
1. Clone or download the repository
2. Open `Housing_Prices_Prediction.ipynb` in your environment
3. Run cells sequentially to execute the analysis

## Key Features
- **Comprehensive EDA**: Exploratory analysis with distribution plots, correlation heatmaps, and pairwise relationships
- **Data Preprocessing**: Handling missing values, categorical encoding, outlier removal using IQR method
- **Feature Selection**: VIF analysis, Recursive Feature Elimination (RFE), and Principal Component Analysis (PCA)
- **Regression Models**: Implements 5 regression models:
  - Multiple Linear Regression (MLR)
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)
  - Elastic-Net Regression (L1 + L2)
  - Polynomial Regression
- **Model Evaluation**: Comprehensive metrics including R² Score, RMSE, MSE, and RSS

## Dataset
**Source**: [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

The dataset contains housing price information with:
- **Target**: `price` - Median value of homes (in $1Ms)
- **Features**: Mix of numerical and categorical variables
- **Format**: CSV file (automatically downloaded via kagglehub)

## Results
The project compares all 5 regression models using evaluation metrics (R², RMSE, MSE, RSS) to identify the best performing algorithm for price prediction. Performance visualizations include:
- R² Score comparison across models
- RMSE comparison for training vs testing sets
- Residual distribution plots
- Actual vs Predicted scatter plots

---

**Reference Notebook**: [Housing Price Prediction - Best ML Algorithms](https://www.kaggle.com/code/yasserh/housing-price-prediction-best-ml-algorithms/notebook)
