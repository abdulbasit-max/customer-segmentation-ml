from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans

from src.preprocessing import build_preprocessing_pipeline

st.set_page_config(
    page_title="Customer Segmentation ML Engine",
    page_icon="📊",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_DIR / "segmented_customers.csv", index_col="CustomerID")
    benchmark = pd.read_csv(PROCESSED_DIR / "model_benchmark.csv")
    return df, benchmark


@st.cache_resource
def fit_live_predictor(df: pd.DataFrame, n_clusters: int):
    feature_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "SkuDiversity",
        "AvgBasketSize",
        "ReturnRate",
    ]
    pipeline = build_preprocessing_pipeline(n_components=0.85)
    X_pca = pipeline.fit_transform(df[feature_cols])
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=42)
    labels = km.fit_predict(X_pca)
    return pipeline, km, X_pca, labels


st.title("Customer Segmentation & Behavioral Clustering Engine")
st.caption(
    "Unsupervised Machine Learning Pipeline | Online Retail II (5,852 Customer Profiles)"
)

df, benchmark_df = load_data()

# Sidebar controls
st.sidebar.header("Model & View Controls")
k_choice = st.sidebar.radio(
    "Select K-Means Granularity:",
    options=[2, 4],
    index=1,
    format_func=lambda k: (
        "k = 2 (Max Silhouette Separation)"
        if k == 2
        else "k = 4 (Granular Business Personas)"
    ),
)

pipeline, km_model, X_pca, live_labels = fit_live_predictor(df, k_choice)
df["Active_Cluster"] = live_labels.astype(str)

# Top KPI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("PCA Variance Retained", "87.09% (3 PCs)")
col3.metric("Active Clusters (k)", f"{k_choice}")
col4.metric("Median Customer Spend", f"£{df['Monetary'].median():,.2f}")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "3D PCA Cluster Explorer",
        "Persona Profiles & Snake Plot",
        "Algorithm Benchmark & Plots",
        "Live Customer Segment Predictor",
    ]
)

with tab1:
    st.subheader(f"3D Principal Component Topology (K-Means, k={k_choice})")
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"], index=df.index)
    pca_df["Cluster"] = df["Active_Cluster"]
    pca_df["Recency"] = df["Recency"]
    pca_df["Frequency"] = df["Frequency"]
    pca_df["Monetary"] = df["Monetary"]

    fig_3d = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        hover_data=["Recency", "Frequency", "Monetary"],
        opacity=0.75,
        height=600,
    )
    fig_3d.update_traces(marker=dict(size=4))
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.subheader("Cluster Behavioral Summary (Median Original Units)")
    feature_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "SkuDiversity",
        "AvgBasketSize",
        "ReturnRate",
    ]
    summary = df.groupby("Active_Cluster")[feature_cols].median()
    summary["Customer_Count"] = df.groupby("Active_Cluster").size()
    summary["Share (%)"] = (summary["Customer_Count"] / len(df) * 100).round(1)
    st.dataframe(summary.round(2), use_container_width=True)

    st.subheader("Interactive Behavioral Snake Plot (Standardized Z-Scores)")
    log_df = np.log1p(df[feature_cols])
    z_df = (log_df - log_df.mean()) / log_df.std()
    z_df["Cluster"] = df["Active_Cluster"]
    snake_means = z_df.groupby("Cluster").mean().reset_index()

    fig_snake = go.Figure()
    for _, row in snake_means.iterrows():
        fig_snake.add_trace(
            go.Scatter(
                x=feature_cols,
                y=[row[col] for col in feature_cols],
                mode="lines+markers",
                name=f"Cluster {row['Cluster']}",
            )
        )
    fig_snake.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_snake.update_layout(
        yaxis_title="Standardized Z-Score (Log Scale)",
        xaxis_title="Behavioral Metric",
        height=450,
    )
    st.plotly_chart(fig_snake, use_container_width=True)

with tab3:
    st.subheader("Internal Validation Benchmark (K-Means vs. GMM vs. DBSCAN)")
    st.dataframe(benchmark_df, use_container_width=True)

    c1, c2 = st.columns(2)
    if (FIGURES_DIR / "kmeans_selection.png").exists():
        c1.image(
            str(FIGURES_DIR / "kmeans_selection.png"),
            caption="K-Means Elbow & Silhouette Curve",
        )
    if (FIGURES_DIR / "gmm_bic_selection.png").exists():
        c2.image(
            str(FIGURES_DIR / "gmm_bic_selection.png"),
            caption="GMM BIC Model Selection",
        )

with tab4:
    st.subheader("Predict Segment for a New or Custom Customer Profile")
    st.write(
        "Adjust the behavioral sliders below to pass a customer vector through the trained `log1p -> RobustScaler -> PCA -> K-Means` pipeline:"
    )

    in1, in2, in3 = st.columns(3)
    recency_in = in1.number_input("Recency (days since last order)", 1, 750, 30)
    freq_in = in2.number_input("Frequency (total unique orders)", 1, 250, 8)
    mon_in = in3.number_input("Monetary (total spend £)", 1.0, 300000.0, 2500.0)

    in4, in5, in6 = st.columns(3)
    sku_in = in4.number_input("SKU Diversity (unique items)", 1, 2000, 95)
    basket_in = in5.number_input("Avg Basket Size (units/order)", 1.0, 5000.0, 200.0)
    ret_in = in6.slider("Return Rate (0.0 to 1.0)", 0.0, 1.0, 0.05, 0.01)

    sample = pd.DataFrame(
        [
            {
                "Recency": recency_in,
                "Frequency": freq_in,
                "Monetary": mon_in,
                "SkuDiversity": sku_in,
                "AvgBasketSize": basket_in,
                "ReturnRate": ret_in,
            }
        ]
    )
    sample_pca = pipeline.transform(sample)
    pred_cluster = km_model.predict(sample_pca)[0]

    st.success(f"**Assigned Segment:** Cluster **{pred_cluster}** (under k={k_choice})")
    st.write(
        f"**Projected PCA Coordinates:** `PC1 = {sample_pca[0, 0]:.3f}`, `PC2 = {sample_pca[0, 1]:.3f}`, `PC3 = {sample_pca[0, 2]:.3f}`"
    )