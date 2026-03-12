# California Housing Price Analysis and Prediction Dashboard

An interactive machine learning dashboard built with Python, Scikit-Learn, and Streamlit to analyze and predict housing prices using the California Housing dataset.

This project combines exploratory data analysis (EDA), regression modeling, model diagnostics, and mathematical explanations in an interactive web application.

--------------------------------------------------

PROJECT OVERVIEW

The goal of this project is to explore the relationship between socioeconomic and geographic variables and housing prices and build predictive models to estimate housing values.

The dashboard allows users to:

- Explore the dataset
- Visualize feature relationships
- Train multiple machine learning models
- Compare model performance
- Inspect model diagnostics
- Export predictions

--------------------------------------------------

DATASET

Source:
Scikit-learn California Housing dataset

Features:

MedInc      : Median income in the block group
HouseAge    : Median house age
AveRooms    : Average number of rooms
AveBedrms   : Average number of bedrooms
Population  : Block group population
AveOccup    : Average household occupancy
Latitude    : Geographic latitude
Longitude   : Geographic longitude

Target variable:

Target      : Median house value proxy used for prediction

--------------------------------------------------

MODELS IMPLEMENTED

1. Multiple Linear Regression
   Baseline regression model using ordinary least squares.

2. Ridge Regression
   Regularized linear regression with an L2 penalty to reduce variance.

3. Random Forest Regressor
   Ensemble model using multiple decision trees to capture nonlinear relationships.

--------------------------------------------------

MODEL EVALUATION METRICS

RMSE (Root Mean Squared Error)
Measures prediction error while penalizing larger errors more strongly.

MAE (Mean Absolute Error)
Average magnitude of prediction errors.

R² (Coefficient of Determination)
Measures the proportion of variance in the target variable explained by the model.

--------------------------------------------------

DASHBOARD FEATURES

Dataset Overview
- Dataset size and structure
- Column information
- Missing value inspection
- Summary statistics

Exploratory Data Analysis
- Histograms of important variables
- Scatter plots of feature relationships
- Correlation matrix
- Feature correlations with the target variable

Model Comparison
- RMSE, MAE, and R² comparison across models
- Actual vs predicted plots
- Residual diagnostics

Model Interpretability
- Linear regression coefficient tables
- Coefficient magnitude visualization
- Random forest feature importance

Diagnostics and Export
- Prediction tables
- Residual summaries
- CSV export of predictions

--------------------------------------------------

PROJECT STRUCTURE

housing-price-ml/

app.py
Main Streamlit dashboard application.

requirements.txt
Python dependencies.

README.txt
Project documentation.

LICENSE
Repository license.

--------------------------------------------------

INSTALLATION

Clone the repository

git clone https://github.com/YOUR_USERNAME/housing-price-ml.git

Navigate to the project directory

cd housing-price-ml

Install dependencies

pip install -r requirements.txt

--------------------------------------------------

RUN THE APPLICATION

Start the Streamlit dashboard

python -m streamlit run app.py

The application will open in your browser at

http://localhost:8501

--------------------------------------------------

TECHNOLOGIES USED

Python  
Streamlit  
Scikit-Learn  
Pandas  
NumPy  
Matplotlib

--------------------------------------------------

FUTURE IMPROVEMENTS

Possible extensions include

- Geographic visualization of housing prices
- Cross-validation for model stability
- Hyperparameter tuning
- Additional models such as Gradient Boosting or XGBoost
- Deployment to a public cloud platform