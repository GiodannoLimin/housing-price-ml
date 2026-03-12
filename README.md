# California Housing Price Prediction Dashboard

An interactive machine learning dashboard built with Python, Scikit-learn, and Streamlit to analyze and predict housing values using the California Housing benchmark dataset.

This project combines exploratory data analysis (EDA), regression modeling, model diagnostics, model interpretation, and mathematical explanations in an interactive web application.

--------------------------------------------------

PROJECT OVERVIEW

The goal of this project is to examine how socioeconomic and geographic variables relate to housing values and to build predictive models for regression analysis.

The dashboard allows users to:

- Explore the dataset
- Visualize feature relationships
- Compare multiple regression models
- Inspect model diagnostics
- Interpret model outputs
- Generate and export predictions

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

Note:
This dataset is a standard benchmark dataset for regression modeling and is used here for model comparison and dashboard development.

--------------------------------------------------

MODELS IMPLEMENTED

1. Multiple Linear Regression
   Baseline regression model using ordinary least squares.

2. Ridge Regression
   Regularized linear regression with an L2 penalty to reduce coefficient variance.

3. Random Forest Regressor
   Tree-based ensemble model used to capture nonlinear relationships between features and the target.

--------------------------------------------------

MODEL EVALUATION METRICS

RMSE (Root Mean Squared Error)
Measures prediction error while penalizing larger errors more strongly.

MAE (Mean Absolute Error)
Measures the average absolute magnitude of prediction errors.

R² (Coefficient of Determination)
Measures the proportion of variance in the target explained by the model.

--------------------------------------------------

DASHBOARD FEATURES

Dataset Overview
- Dataset size and structure
- Column information
- Missing value inspection
- Summary statistics

Exploratory Data Analysis
- Histograms of key variables
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

Prediction Tool
- Interactive feature inputs
- Real-time model prediction output

Diagnostics and Export
- Prediction tables
- Residual summaries
- CSV export of predictions

--------------------------------------------------

PROJECT STRUCTURE

housing-price-ml/

app.py
Main Streamlit dashboard application

requirements.txt
Python dependencies

README.md
Project documentation

LICENSE
Repository license

--------------------------------------------------

INSTALLATION

Clone the repository

git clone https://github.com/GiodannoLimin/housing-price-ml.git

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
Scikit-learn
Pandas
NumPy
Matplotlib

--------------------------------------------------

LIMITATIONS

- This project uses a benchmark dataset and does not represent current housing market conditions.
- Model performance depends on the train-test split and selected hyperparameters.
- The target variable is a dataset-defined housing value proxy rather than a real-time appraisal output.

--------------------------------------------------

FUTURE IMPROVEMENTS

Possible extensions include:

- Geographic visualization of housing prices
- Cross-validation for model stability
- Hyperparameter tuning
- Additional models such as Gradient Boosting or XGBoost
- Deployment to a public cloud platform