import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error


st.set_page_config(page_title="California Housing Dashboard", layout="wide")

plt.style.use("dark_background")

PLOT_FACE = "#0E1117"
AXIS_FACE = "#111827"
GRID_ALPHA = 0.18
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE = 9

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_facecolor(AXIS_FACE)
    ax.grid(True, alpha=GRID_ALPHA)
# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    df = df.rename(columns={"MedHouseVal": "Target"})
    return df


# -----------------------------
# Modeling
# -----------------------------
@st.cache_resource
def train_models():
    df = load_data()

    X = df.drop(columns=["Target"])
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    results["Linear Regression"] = {
        "model": lr,
        "predictions": lr_pred,
        "y_test": y_test.reset_index(drop=True),
        "rmse": root_mean_squared_error(y_test, lr_pred),
        "mae": mean_absolute_error(y_test, lr_pred),
        "r2": r2_score(y_test, lr_pred),
        "coefficients": pd.DataFrame(
            {"Feature": X.columns, "Coefficient": lr.coef_}
        ).sort_values(by="Coefficient", key=abs, ascending=False),
    }

    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    results["Ridge Regression"] = {
        "model": ridge,
        "predictions": ridge_pred,
        "y_test": y_test.reset_index(drop=True),
        "rmse": root_mean_squared_error(y_test, ridge_pred),
        "mae": mean_absolute_error(y_test, ridge_pred),
        "r2": r2_score(y_test, ridge_pred),
        "coefficients": pd.DataFrame(
            {"Feature": X.columns, "Coefficient": ridge.coef_}
        ).sort_values(by="Coefficient", key=abs, ascending=False),
    }

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
        max_depth=12,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    results["Random Forest"] = {
        "model": rf,
        "predictions": rf_pred,
        "y_test": y_test.reset_index(drop=True),
        "rmse": root_mean_squared_error(y_test, rf_pred),
        "mae": mean_absolute_error(y_test, rf_pred),
        "r2": r2_score(y_test, rf_pred),
        "importance": pd.DataFrame(
            {"Feature": X.columns, "Importance": rf.feature_importances_}
        ).sort_values(by="Importance", ascending=False),
    }

    comparison = pd.DataFrame(
        [
            {
                "Model": model_name,
                "RMSE": info["rmse"],
                "MAE": info["mae"],
                "R²": info["r2"],
            }
            for model_name, info in results.items()
        ]
    ).sort_values(by="RMSE")

    return df, X, y, results, comparison


# -----------------------------
# Plot helpers
# -----------------------------
def plot_histogram(series, title, xlabel):
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    fig.patch.set_facecolor(PLOT_FACE)
    ax.hist(series, bins=30, alpha=0.85)
    style_ax(ax, title, xlabel, "Frequency")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

def plot_scatter(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    fig.patch.set_facecolor(PLOT_FACE)
    ax.scatter(x, y, alpha=0.25, s=12)
    style_ax(ax, title, xlabel, ylabel)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    fig.patch.set_facecolor(PLOT_FACE)
    im = ax.imshow(corr, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_facecolor(AXIS_FACE)
    ax.set_title("Correlation Matrix", fontsize=TITLE_SIZE, pad=10)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def plot_actual_vs_predicted(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    fig.patch.set_facecolor(PLOT_FACE)
    ax.scatter(y_true, y_pred, alpha=0.25, s=12)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.5)

    style_ax(ax, title, "Actual Target", "Predicted Target")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    fig.patch.set_facecolor(PLOT_FACE)
    ax.scatter(y_pred, residuals, alpha=0.25, s=12)
    ax.axhline(y=0, linestyle="--", linewidth=1.5)

    style_ax(ax, title, "Predicted Target", "Residual")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def plot_feature_bar(df_plot, value_col, title):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    fig.patch.set_facecolor(PLOT_FACE)

    plot_df = df_plot.copy().iloc[::-1]
    ax.barh(plot_df["Feature"], plot_df[value_col], alpha=0.9)

    style_ax(ax, title, value_col, "")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


# -----------------------------
# Main app
# -----------------------------
df, X, y, model_results, comparison_df = train_models()

st.title("California Housing Price Analysis and Prediction Dashboard")
st.markdown(
    """
This app combines **exploratory data analysis**, **regression modeling**, and
**model diagnostics** for the California housing dataset.

The target variable is the median house value proxy used in the dataset.
"""
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "EDA", "Math", "Models", "Diagnostics & Download"]
)

# -----------------------------
# Tab 1: Overview
# -----------------------------
with tab1:
    st.header("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Features", f"{df.shape[1] - 1}")

    st.subheader("First 5 rows")
    st.dataframe(df.head())

    st.subheader("Column names")
    st.write(list(df.columns))

    st.subheader("Data types and missing values")
    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Dtype": [str(df[col].dtype) for col in df.columns],
            "Missing Values": [int(df[col].isnull().sum()) for col in df.columns],
        }
    )
    st.dataframe(info_df)

    st.subheader("Summary statistics")
    st.dataframe(df.describe())

    st.subheader("Interpretation of variables")
    st.markdown(
        """
- **MedInc**: median income in the block group  
- **HouseAge**: median house age  
- **AveRooms**: average number of rooms  
- **AveBedrms**: average number of bedrooms  
- **Population**: block group population  
- **AveOccup**: average household occupancy  
- **Latitude / Longitude**: geographic location  
- **Target**: response variable to predict
"""
    )


# -----------------------------
# Tab 2: EDA
# -----------------------------
with tab2:
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        plot_histogram(df["Target"], "Distribution of Target", "Target")
    with col2:
        plot_histogram(df["MedInc"], "Distribution of Median Income", "MedInc")

    col3, col4 = st.columns(2)
    with col3:
        plot_scatter(
            df["MedInc"],
            df["Target"],
            "Median Income vs Target",
            "MedInc",
            "Target",
        )
    with col4:
        plot_scatter(
            df["HouseAge"],
            df["Target"],
            "House Age vs Target",
            "HouseAge",
            "Target",
        )

    st.subheader("Correlation matrix")
    plot_correlation_heatmap(df)

    st.subheader("Top correlations with Target")
    corr_target = (
        df.corr(numeric_only=True)["Target"]
        .drop("Target")
        .sort_values(key=np.abs, ascending=False)
        .reset_index()
    )
    corr_target.columns = ["Feature", "Correlation with Target"]
    st.dataframe(corr_target)


# -----------------------------
# Tab 3: Math
# -----------------------------
with tab3:
    st.header("Mathematical Formulation")

    st.markdown("### Multiple Linear Regression")
    st.latex(
        r"""
        y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i
        """
    )

    st.markdown(
        """
In matrix form:
"""
    )
    st.latex(
        r"""
        \mathbf{y} = \mathbf{X}\beta + \varepsilon
        """
    )

    st.markdown(
        """
The ordinary least squares estimator minimizes the residual sum of squares:
"""
    )
    st.latex(
        r"""
        \hat{\beta}
        =
        \arg\min_{\beta}
        \|\mathbf{y} - \mathbf{X}\beta\|_2^2
        """
    )

    st.markdown("### Ridge Regression")
    st.latex(
        r"""
        \hat{\beta}_{ridge}
        =
        \arg\min_{\beta}
        \left(
        \|\mathbf{y} - \mathbf{X}\beta\|_2^2
        + \lambda \|\beta\|_2^2
        \right)
        """
    )

    st.markdown(
    """
    Ridge regression adds an **L₂ penalty**, which shrinks coefficients and can reduce variance.
    """
    )

    st.markdown("### Random Forest")
    st.markdown(
        """
A random forest is an ensemble of decision trees. For regression, the prediction is the average:
"""
    )
    st.latex(
        r"""
        \hat{f}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x)
        """
    )

    st.markdown(
    r"""
    where $T_b(x)$ is the prediction from tree $b$.
    """
    )
    st.markdown("### Metrics")
    st.latex(
        r"""
        \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
        """
    )
    st.latex(
        r"""
        \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
        """
    )
    st.latex(
        r"""
        R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
        """
    )

    st.markdown(
    """
    - **RMSE** penalizes large errors more strongly  
    - **MAE** is easier to interpret in absolute-error terms  
    - **R²** measures the proportion of variance explained
    """
    )


# -----------------------------
# Tab 4: Models
# -----------------------------
with tab4:
    st.header("Model Comparison")

    st.dataframe(
    comparison_df.reset_index(drop=True).style.format(
        {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}"}
    ),
    use_container_width=True,
    )

    model_choice = st.selectbox(
        "Choose a model",
        ["Linear Regression", "Ridge Regression", "Random Forest"],
    )

    results = model_results[model_choice]
    y_test = results["y_test"]
    predictions = results["predictions"]

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{results['rmse']:.4f}")
    c2.metric("MAE", f"{results['mae']:.4f}")
    c3.metric("R²", f"{results['r2']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        plot_actual_vs_predicted(y_test, predictions, f"{model_choice}: Actual vs Predicted")
    with col2:
        plot_residuals(y_test, predictions, f"{model_choice}: Residual Plot")

    if model_choice in ["Linear Regression", "Ridge Regression"]:
        st.subheader("Coefficient Table")
        st.dataframe(results["coefficients"].reset_index(drop=True), use_container_width=True)

        st.subheader("Coefficient Magnitudes")
        coef_plot_df = results["coefficients"].copy()
        coef_plot_df["Coefficient"] = coef_plot_df["Coefficient"].astype(float)
        plot_feature_bar(coef_plot_df.head(6), "Coefficient", f"{model_choice}: Top Coefficients")

        st.markdown(
            """
For linear models, a coefficient estimates the expected marginal change in the target
for a one-unit increase in that feature, holding the other features fixed.
"""
        )

    if model_choice == "Random Forest":
        st.subheader("Feature Importance")
        st.dataframe(results["importance"].reset_index(drop=True), use_container_width=True)

        plot_feature_bar(
            results["importance"].head(6),
            "Importance",
            "Random Forest: Top Feature Importances",
        )

        st.markdown(
            """
Feature importance here reflects how much a variable contributes to reducing prediction error
across the ensemble of trees.
"""
        )


# -----------------------------
# Tab 5: Diagnostics & Download
# -----------------------------
with tab5:
    st.header("Diagnostics and Export")

    selected_model = st.selectbox(
        "Choose model for downloadable predictions",
        ["Linear Regression", "Ridge Regression", "Random Forest"],
        key="download_model",
    )

    selected_results = model_results[selected_model]
    diag_df = pd.DataFrame(
        {
            "Actual": selected_results["y_test"],
            "Predicted": selected_results["predictions"],
        }
    )
    diag_df["Residual"] = diag_df["Actual"] - diag_df["Predicted"]
    diag_df["Absolute Error"] = np.abs(diag_df["Residual"])

    st.subheader("Prediction sample")
    st.dataframe(diag_df.head(25))

    st.subheader("Residual summary")
    st.dataframe(diag_df[["Residual", "Absolute Error"]].describe())

    csv = diag_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv,
        file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )

    st.markdown(
        r"""
### Notes
- The dashed line in the actual-vs-predicted plot is **not** the model itself.  
- It is the line $\hat{y} = y$, which represents perfect prediction.  
- A model is better when points concentrate more tightly around that diagonal.
"""
    )