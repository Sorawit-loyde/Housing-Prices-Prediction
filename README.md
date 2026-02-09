# Housing Prices Prediction

## Project Overview

This project implements a comprehensive machine learning pipeline to predict housing prices using various regression techniques. The analysis focuses on exploratory data analysis, feature engineering, and comparing multiple regression models to identify the best performing algorithm for price prediction.

The project follows a structured approach from data loading and preprocessing through to model evaluation and comparison, utilizing techniques like feature scaling, feature selection, and regularization methods.

## Dataset

**Source:** [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

The dataset contains housing price information with various features including numerical and categorical variables. The data encompasses multiple attributes that influence housing prices, making it suitable for regression analysis and predictive modeling.

### Dataset Characteristics
- **Target Variable:** `price` - Median value of homes (in $1Ms)
- **Features:** Mix of numerical and categorical features
- **Size:** Multiple samples with comprehensive housing attributes
- **Format:** CSV file

## Key Features

### 1. **Exploratory Data Analysis (EDA)**
- Target variable distribution analysis
- Categorical features visualization using countplots
- Numerical features distribution with KDE plots
- Box plots for outlier identification
- Pairwise relationships visualization
- Correlation matrix heatmap for feature relationships

### 2. **Data Preprocessing**
- Duplicate value detection and removal
- Missing data analysis and handling
- Categorical feature encoding:
  - One-hot encoding for binary features
  - Dummy encoding for multi-category features
- Outlier removal using IQR (Interquartile Range) method
- Feature scaling using standardization (mean=0, std=1)

### 3. **Feature Selection & Extraction**
- **Variance Inflation Factor (VIF):** Identifies and removes multicollinear features
- **Recursive Feature Elimination (RFE):** Automatically selects optimal number of features
- **Principal Component Analysis (PCA):** Reduces dimensionality while preserving variance
- Correlation analysis for target variable relationships

### 4. **Regression Models Implemented**
- **Multiple Linear Regression (MLR):** Baseline model using all features
- **Ridge Regression:** L2 regularization to prevent overfitting
- **Lasso Regression:** L1 regularization for feature selection
- **Elastic-Net Regression:** Combination of L1 and L2 regularization
- **Polynomial Regression:** Higher-degree polynomial features for non-linear relationships

### 5. **Model Evaluation Metrics**
- **R² Score:** Coefficient of determination
- **RSS (Residual Sum of Squares):** Total prediction error
- **MSE (Mean Squared Error):** Average squared prediction error
- **RMSE (Root Mean Squared Error):** Scale-independent error metric
- Residual distribution analysis
- Actual vs. Predicted scatter plots

## Dependencies

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical operations and array processing
- **scikit-learn** - Machine learning algorithms and preprocessing
- **matplotlib** - Data visualization and plotting
- **seaborn** - Statistical visualizations and heatmaps
- **statsmodels** - Statistical modeling and analysis
- **kagglehub** - Kaggle dataset downloading

### Installation

Install all dependencies using pip:

```bash
pip install pandas matplotlib seaborn numpy scikit-learn kagglehub statsmodels
```

Or using the requirements approach:

```bash
pip install pandas matplotlib seaborn numpy scikit-learn kagglehub statsmodels scipy
```

## Project Structure

```
Housing-Prices-Prediction/
├── Housing_Prices_Prediction.ipynb  # Main analysis notebook
├── Housing.csv                       # Dataset (downloaded from Kaggle)
└── README.md                         # Project documentation
```

## How to Use

### Option 1: Local Jupyter Notebook

1. **Clone or download the project files**
   ```bash
   cd Housing-Prices-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas matplotlib seaborn numpy scikit-learn kagglehub statsmodels
   ```

3. **Start Jupyter Notebook**
   ```bash
   jupyter notebook Housing_Prices_Prediction.ipynb
   ```

4. **Run all cells** or follow the step-by-step analysis

### Option 2: Google Colab (Recommended for Quick Access)

Access the pre-configured Google Colab notebook here:
[Housing Prices Prediction - Google Colab](https://colab.research.google.com/drive/1ZY2STt_2uzzvB24JrNRzJaAMkET4O7x_?usp=sharing)

No installation required - simply open and run cells directly in your browser!

### Option 3: Reference Kaggle Notebook

For additional implementations and techniques, refer to the original Kaggle notebook:
[Housing Price Prediction - Best ML Algorithms](https://www.kaggle.com/code/yasserh/housing-price-prediction-best-ml-algorithms/notebook)

## Workflow Overview

### Step 1: Data Loading and Import
- Load dataset from Kaggle using `kagglehub`
- Import all required libraries
- Initial data exploration

### Step 2: Exploratory Data Analysis (EDA)
- Analyze data types, unique values, and missing data
- Identify numerical vs. categorical features
- Visualize distributions and relationships

### Step 3: Data Preprocessing
- Remove duplicates and handle missing values
- Encode categorical variables
- Remove outliers using IQR method
- Split data into training (80%) and testing (20%) sets

### Step 4: Feature Engineering
- Standardize numerical features
- Apply feature selection techniques:
  - VIF analysis for multicollinearity
  - RFE for automatic feature selection
  - PCA for dimensionality reduction

### Step 5: Model Development
- Train multiple regression models
- Compare performance on train and test datasets
- Analyze model coefficients and predictions

### Step 6: Model Evaluation & Comparison
- Calculate evaluation metrics (R², RMSE, MSE, RSS)
- Compare all models using consistent metrics
- Visualize performance comparisons
- Identify best performing model

## Model Performance

The notebook evaluates the following models:

| Model | Use Case |
|-------|----------|
| Multiple Linear Regression | Baseline with all features |
| Ridge Regression | When multicollinearity is present |
| Lasso Regression | Feature selection through regularization |
| Elastic-Net | Combines Ridge and Lasso benefits |
| Polynomial Regression | Non-linear relationship handling |

Results are compared using R² scores and RMSE metrics to determine the optimal model for price prediction.

## Key Findings

- The preprocessing steps significantly impact model performance
- Feature scaling is crucial for regression algorithms
- Regularization techniques help prevent overfitting
- Polynomial features can improve model fit but may risk overfitting
- Comprehensive feature selection improves model generalization

## Techniques Highlighted

1. **Data Cleaning:** Handling duplicates, missing values, and outliers
2. **Feature Encoding:** One-hot and dummy encoding for categorical variables
3. **Standardization:** Zero-mean, unit-variance scaling
4. **Feature Selection:** VIF, RFE, and PCA methods
5. **Regularization:** Ridge, Lasso, and Elastic-Net penalties
6. **Model Evaluation:** Multiple metrics for comprehensive assessment
7. **Visualization:** Distribution plots, correlation heatmaps, residual analysis

## Files Description

- **Housing_Prices_Prediction.ipynb** - Complete analysis notebook with all steps and visualizations
- **Housing.csv** - Housing prices dataset (automatically downloaded from Kaggle)
- **README.md** - This documentation file

## Requirements

- Python 3.7+
- Jupyter Notebook or Google Colab
- All dependencies listed in the Dependencies section
- Kaggle account (for direct dataset access)

## Getting Started

1. Open the Jupyter notebook or access Google Colab
2. Run the dependency installation cell
3. Execute cells sequentially to understand the workflow
4. Modify parameters to experiment with different techniques
5. Visualize results and compare model performance

## Troubleshooting

**Issue:** Kaggle authentication error when downloading dataset
- **Solution:** Configure Kaggle API credentials following [Kaggle API Setup](https://www.kaggle.com/settings/account)

**Issue:** Memory issues with large datasets
- **Solution:** Use Google Colab (free GPU/RAM available) or reduce dataset size

**Issue:** Missing library errors
- **Solution:** Reinstall dependencies: `pip install --upgrade pandas scikit-learn matplotlib seaborn statsmodels`

## Future Enhancements

- Cross-validation for more robust model evaluation
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Ensemble methods (Random Forest, Gradient Boosting)
- Deep learning approaches (Neural Networks)
- Feature importance analysis
- Production model deployment

## References

- [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
- [Original Kaggle Notebook](https://www.kaggle.com/code/yasserh/housing-price-prediction-best-ml-algorithms/notebook)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Statsmodels Documentation](https://www.statsmodels.org/)

## Author

This project demonstrates comprehensive machine learning workflow from data exploration to model evaluation, following best practices in data science and machine learning engineering.

## License

Feel free to use this project for learning and educational purposes.

## Acknowledgments

- Dataset source: [Yasser H. on Kaggle](https://www.kaggle.com/yasserh)
- Original notebook inspiration from Kaggle community
- Special thanks to the open-source data science community

---

**Last Updated:** February 2026

For questions or improvements, refer to the original sources and documentation linked above.
