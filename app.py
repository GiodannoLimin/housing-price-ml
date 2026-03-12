import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="California Housing Price Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.style.use("dark_background")

# =========================================================
# Styling
# =========================================================
PLOT_FACE = "#0d1322"
AXIS_FACE = "#111827"
GRID_ALPHA = 0.15
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE = 9

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1330 45%, #1b1f45 100%);
        color: white;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(168,85,247,0.10));
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 22px;
        padding: 1.5rem 1.5rem 1.2rem 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }

    .subtle-text {
        color: #cbd5e1;
        font-size: 0.98rem;
        line-height: 1.75;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 16px;
        border-radius: 18px;
        backdrop-filter: blur(6px);
    }

    [data-testid="stSidebar"] {
        background: rgba(8, 12, 30, 0.92);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stTabs"] button {
        color: #cbd5e1;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: white;
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    div[data-testid="stDownloadButton"] > button {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_facecolor(AXIS_FACE)
    ax.grid(True, alpha=GRID_ALPHA)


# =========================================================
# Data loading
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    df = df.rename(columns={"MedHouseVal": "Target"})
    return df


# =========================================================
# Modeling
# =========================================================
@st.cache_resource(show_spinner=False)
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
        n_estimators=80,
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

    return df, X, y, X_train, X_test, y_train, y_test, results, comparison


# =========================================================
# Plot helpers
# =========================================================
def plot_histogram(series, title, xlabel):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    fig.patch.set_facecolor(PLOT_FACE)
    ax.hist(series, bins=30, alpha=0.9)
    style_ax(ax, title, xlabel, "Frequency")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def plot_scatter(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    fig.patch.set_facecolor(PLOT_FACE)
    ax.scatter(x, y, alpha=0.25, s=12)
    style_ax(ax, title, xlabel, ylabel)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
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
    st.pyplot(fig, width="stretch")


def plot_actual_vs_predicted(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    fig.patch.set_facecolor(PLOT_FACE)
    ax.scatter(y_true, y_pred, alpha=0.25, s=12)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.5)

    style_ax(ax, title, "Actual Target", "Predicted Target")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    fig.patch.set_facecolor(PLOT_FACE)
    ax.scatter(y_pred, residuals, alpha=0.25, s=12)
    ax.axhline(y=0, linestyle="--", linewidth=1.5)

    style_ax(ax, title, "Predicted Target", "Residual")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def plot_feature_bar(df_plot, value_col, title):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    fig.patch.set_facecolor(PLOT_FACE)

    plot_df = df_plot.copy().iloc[::-1]
    ax.barh(plot_df["Feature"], plot_df[value_col], alpha=0.9)

    style_ax(ax, title, value_col, "")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def plot_model_metric_bar(df_metrics, metric):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    fig.patch.set_facecolor(PLOT_FACE)
    ax.bar(df_metrics["Model"], df_metrics[metric], alpha=0.9)
    style_ax(ax, f"Model Comparison: {metric}", "Model", metric)
    plt.xticks(rotation=12)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


# =========================================================
# Load data / compute summary
# =========================================================
loading = st.empty()

loading.markdown(
    """
    <div style="
        display:flex;
        justify-content:center;
        align-items:center;
        height:140px;
        font-size:18px;
        font-weight:500;
        color:#cbd5e1;
    ">
        Initializing models and preparing dashboard<span class="dot"></span>
    </div>

    <style>
    .dot::after{
        content:'';
        animation: dots 1.4s infinite;
    }

    @keyframes dots {
        0% {content:'';}
        25% {content:'.';}
        50% {content:'..';}
        75% {content:'...';}
        100% {content:'';}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

df, X, y, X_train, X_test, y_train, y_test, model_results, comparison_df = train_models()

loading.empty()

best_model_row = comparison_df.iloc[0]
best_model_name = best_model_row["Model"]
best_rmse = best_model_row["RMSE"]
best_mae = best_model_row["MAE"]
best_r2 = best_model_row["R²"]

corr_target = (
    df.corr(numeric_only=True)["Target"]
    .drop("Target")
    .sort_values(key=np.abs, ascending=False)
    .reset_index()
)
corr_target.columns = ["Feature", "Correlation with Target"]
top_feature = corr_target.iloc[0]["Feature"]
top_corr = corr_target.iloc[0]["Correlation with Target"]


# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("Dashboard Controls")

selected_model = st.sidebar.selectbox(
    "Select model",
    ["Linear Regression", "Ridge Regression", "Random Forest"],
)

show_raw_data = st.sidebar.checkbox("Show raw dataset preview", value=False)
show_math = st.sidebar.checkbox("Show mathematical formulation", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Tech stack**  
Python · Streamlit · scikit-learn · Pandas · Matplotlib
"""
)

st.sidebar.markdown(
    """
**Dataset note**  
This project uses the California Housing benchmark dataset, a standard regression dataset
containing socioeconomic and geographic variables.
"""
)


# =========================================================
# Hero section
# =========================================================
st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom:0.35rem;">California Housing Price Prediction Dashboard</h1>
        <p class="subtle-text">
            Interactive machine learning dashboard for regression analysis on the California Housing
            benchmark dataset. The app combines exploratory data analysis, model comparison,
            diagnostic plots, feature interpretation, and an interactive prediction tool.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Observations", f"{df.shape[0]:,}")
metric_col2.metric("Features", f"{df.shape[1] - 1}")
metric_col3.metric("Best Model", best_model_name)
metric_col4.metric("Best RMSE", f"{best_rmse:.4f}")

st.markdown(
    f"""
This dashboard examines how **socioeconomic and geographic variables** relate to housing values.
Across the evaluated models, **{best_model_name}** achieved the strongest performance on the test set
based on RMSE. The strongest absolute linear correlation with the target is **{top_feature}**
at **{top_corr:.3f}**, suggesting that it is an important predictor in this benchmark setting.
"""
)

if show_raw_data:
    with st.expander("Preview raw dataset"):
        st.dataframe(df.head(20), width="stretch")

st.info("Use the sidebar on the left to change models, show the raw dataset, and toggle the mathematical section.")


# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "EDA", "Models", "Prediction Tool", "Diagnostics & Export"]
)


# =========================================================
# Tab 1: Overview
# =========================================================
with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Train / Test Split", "80% / 20%")

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### Variable Summary")
        info_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Dtype": [str(df[col].dtype) for col in df.columns],
                "Missing Values": [int(df[col].isnull().sum()) for col in df.columns],
            }
        )
        st.dataframe(info_df, width="stretch")

    with right:
        st.markdown("### Variable Interpretation")
        st.markdown(
            """
- **MedInc**: median income in the block group  
- **HouseAge**: median house age  
- **AveRooms**: average rooms per household  
- **AveBedrms**: average bedrooms per household  
- **Population**: block group population  
- **AveOccup**: average household occupancy  
- **Latitude / Longitude**: geographic location  
- **Target**: housing value proxy used as the response variable
"""
        )

    st.markdown("### Summary Statistics")
    st.dataframe(df.describe(), width="stretch")

    st.markdown("### Dataset Framing")
    st.markdown(
        """
The California Housing dataset is a **widely used benchmark dataset for regression modeling**.
It is useful for demonstrating supervised learning workflows, model diagnostics, and feature-based
interpretation. This project is intended as a modeling and dashboarding exercise rather than a
real-time housing market appraisal tool.
"""
    )

    if show_math:
        st.markdown("### Mathematical Formulation")

        st.markdown("**Multiple Linear Regression**")
        st.latex(
            r"""
            y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i
            """
        )
        st.latex(r"""\mathbf{y} = \mathbf{X}\beta + \varepsilon""")
        st.latex(
            r"""
            \hat{\beta}
            =
            \arg\min_{\beta}
            \|\mathbf{y} - \mathbf{X}\beta\|_2^2
            """
        )

        st.markdown("**Ridge Regression**")
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

        st.markdown("**Random Forest Regression**")
        st.latex(
            r"""
            \hat{f}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x)
            """
        )

        st.markdown("**Evaluation Metrics**")
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


# =========================================================
# Tab 2: EDA
# =========================================================
with tab2:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        plot_histogram(df["Target"], "Distribution of Target", "Target")
    with col2:
        plot_histogram(df["MedInc"], "Distribution of Median Income", "MedInc")

    col3, col4 = st.columns(2)
    with col3:
        plot_scatter(
            df["MedInc"], df["Target"], "Median Income vs Target", "MedInc", "Target"
        )
    with col4:
        plot_scatter(
            df["HouseAge"], df["Target"], "House Age vs Target", "HouseAge", "Target"
        )

    st.markdown("### Correlation Matrix")
    plot_correlation_heatmap(df)

    st.markdown("### Correlations with Target")
    st.dataframe(corr_target, width="stretch")
    st.markdown(
        f"""
The strongest absolute linear association with the target is **{top_feature}**
with correlation **{top_corr:.3f}**. This makes it a strong candidate for importance
in both linear and nonlinear models.
"""
    )


# =========================================================
# Tab 3: Models
# =========================================================
with tab3:
    st.subheader("Model Comparison")

    c1, c2, c3 = st.columns(3)
    c1.metric("Lowest RMSE", f"{best_rmse:.4f}")
    c2.metric("Lowest MAE", f"{best_mae:.4f}")
    c3.metric("Best R²", f"{best_r2:.4f}")

    st.dataframe(
        comparison_df.reset_index(drop=True).style.format(
            {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}"}
        ),
        width="stretch",
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        plot_model_metric_bar(comparison_df, "RMSE")
    with col_b:
        plot_model_metric_bar(comparison_df, "MAE")
    with col_c:
        plot_model_metric_bar(comparison_df, "R²")

    st.markdown("---")
    st.subheader(f"{selected_model} Details")

    results = model_results[selected_model]
    y_test_selected = results["y_test"]
    predictions = results["predictions"]

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{results['rmse']:.4f}")
    c2.metric("MAE", f"{results['mae']:.4f}")
    c3.metric("R²", f"{results['r2']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        plot_actual_vs_predicted(
            y_test_selected, predictions, f"{selected_model}: Actual vs Predicted"
        )
    with col2:
        plot_residuals(
            y_test_selected, predictions, f"{selected_model}: Residual Plot"
        )

    if selected_model in ["Linear Regression", "Ridge Regression"]:
        st.markdown("### Coefficient Analysis")
        st.dataframe(
            results["coefficients"].reset_index(drop=True),
            width="stretch",
        )

        coef_plot_df = results["coefficients"].copy()
        coef_plot_df["Coefficient"] = coef_plot_df["Coefficient"].astype(float)
        plot_feature_bar(
            coef_plot_df.head(6),
            "Coefficient",
            f"{selected_model}: Top Coefficients",
        )

        st.markdown(
            """
For linear models, each coefficient estimates the expected marginal change in the target
for a one-unit increase in that feature, holding the other variables fixed.
"""
        )

    if selected_model == "Random Forest":
        st.markdown("### Feature Importance")
        st.dataframe(
            results["importance"].reset_index(drop=True),
            width="stretch",
        )

        plot_feature_bar(
            results["importance"].head(6),
            "Importance",
            "Random Forest: Top Feature Importances",
        )

        st.markdown(
            """
Feature importance reflects how much each variable contributes to reducing prediction
error across the ensemble. It is useful for interpretation, but it does not imply causality.
"""
        )


# =========================================================
# Tab 4: Prediction Tool
# =========================================================
with tab4:
    st.subheader("Interactive Prediction Tool")
    st.markdown(
        """
Enter feature values below to generate a predicted target value using the selected model.
This tool is designed to demonstrate how different regression models respond to feature inputs.
"""
    )

    default_vals = df.drop(columns=["Target"]).median()

    input_col1, input_col2, input_col3 = st.columns(3)

    with input_col1:
        medinc = st.number_input("Median Income", value=float(default_vals["MedInc"]))
        houseage = st.number_input("House Age", value=float(default_vals["HouseAge"]))
        averooms = st.number_input("Average Rooms", value=float(default_vals["AveRooms"]))

    with input_col2:
        avebedrms = st.number_input(
            "Average Bedrooms", value=float(default_vals["AveBedrms"])
        )
        population = st.number_input("Population", value=float(default_vals["Population"]))
        aveoccup = st.number_input(
            "Average Occupancy", value=float(default_vals["AveOccup"])
        )

    with input_col3:
        latitude = st.number_input("Latitude", value=float(default_vals["Latitude"]))
        longitude = st.number_input("Longitude", value=float(default_vals["Longitude"]))
        predict_model = st.selectbox(
            "Prediction model",
            ["Linear Regression", "Ridge Regression", "Random Forest"],
            key="prediction_model",
        )

    input_df = pd.DataFrame(
        {
            "MedInc": [medinc],
            "HouseAge": [houseage],
            "AveRooms": [averooms],
            "AveBedrms": [avebedrms],
            "Population": [population],
            "AveOccup": [aveoccup],
            "Latitude": [latitude],
            "Longitude": [longitude],
        }
    )

    prediction = model_results[predict_model]["model"].predict(input_df)[0]

    p1, p2 = st.columns(2)
    p1.metric("Selected Model", predict_model)
    p2.metric("Predicted Target Value", f"{prediction:.4f}")

    st.dataframe(input_df, width="stretch")

    st.markdown(
        """
This predicted value corresponds to the target definition used in the California Housing
benchmark dataset. It should be interpreted as a model output for demonstration and
comparison purposes rather than as a current market valuation.
"""
    )


# =========================================================
# Tab 5: Diagnostics & Export
# =========================================================
with tab5:
    st.subheader("Diagnostics and Export")

    selected_results = model_results[selected_model]
    diag_df = pd.DataFrame(
        {
            "Actual": selected_results["y_test"],
            "Predicted": selected_results["predictions"],
        }
    )
    diag_df["Residual"] = diag_df["Actual"] - diag_df["Predicted"]
    diag_df["Absolute Error"] = np.abs(diag_df["Residual"])

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### Prediction Sample")
        st.dataframe(diag_df.head(25), width="stretch")

    with right:
        st.markdown("### Residual Summary")
        st.dataframe(
            diag_df[["Residual", "Absolute Error"]].describe(),
            width="stretch",
        )

    csv = diag_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv,
        file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )

    st.markdown("### Interpretation Notes")
    st.markdown(
        r"""
- The dashed line in the actual-vs-predicted plot represents $\hat{y}=y$, the line of perfect prediction.  
- A stronger predictive fit is indicated when points cluster more closely around that diagonal.  
- Residual plots help assess whether prediction errors are centered near zero and whether large deviations remain.
"""
    )

    st.markdown("### Limitations")
    st.markdown(
        """
- This project uses a benchmark dataset, so it does not represent current market conditions.  
- Model performance depends on the train-test split and selected hyperparameters.  
- Random forest feature importance is helpful for interpretation, but it does not establish causal relationships.  
- The target variable is a dataset-defined housing value proxy rather than a real-time appraisal output.
"""
    )