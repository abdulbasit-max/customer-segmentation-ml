from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


# Non-merchandise stock codes present in Online Retail II
NON_PRODUCT_CODES = {
    "POST",
    "D",
    "C2",
    "M",
    "BANK CHARGES",
    "PADS",
    "DOT",
    "CRUK",
    "ADJUST",
    "ADJUST2",
    "TEST001",
    "TEST002",
}


def load_raw_data(raw_dir: Path) -> pd.DataFrame:
    """Loads and concatenates the 2009-2010 and 2010-2011 CSV files."""
    file_2009 = raw_dir / "Year 2009-2010.csv"
    file_2010 = raw_dir / "Year 2010-2011.csv"

    # Online Retail II CSVs commonly use ISO-8859-1 / latin1 encoding
    df_2009 = pd.read_csv(file_2009, encoding="ISO-8859-1")
    df_2010 = pd.read_csv(file_2010, encoding="ISO-8859-1")

    df = pd.concat([df_2009, df_2010], ignore_index=True)
    # Remove exact duplicate rows across overlapping December 2010 records
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[Load] Combined raw dataset shape (deduplicated): {df.shape}")
    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Executes Phase 1 (Sanitization) and Phase 2 (RFM + Behavioral Features)."""
    # Standardize column names
    df = df.rename(columns={"Customer ID": "CustomerID"})

    # Drop transactions with missing CustomerID
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # --- Calculate Return Rate per customer BEFORE filtering cancellations ---
    total_invoices = df.groupby("CustomerID")["Invoice"].nunique()
    cancelled_mask = df["Invoice"].str.startswith("C") | (df["Quantity"] < 0)
    cancelled_invoices = (
        df[cancelled_mask].groupby("CustomerID")["Invoice"].nunique()
    )
    return_rate = (cancelled_invoices / total_invoices).fillna(0.0)

    # --- Phase 1: Filter valid positive merchandise purchases ---
    valid_df = df[
        (~cancelled_mask)
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & (~df["StockCode"].isin(NON_PRODUCT_CODES))
    ].copy()

    valid_df["TotalSpend"] = valid_df["Quantity"] * valid_df["Price"]
    print(f"[Clean] Valid purchase rows retained: {valid_df.shape[0]:,}")

    # --- Phase 2: Aggregate to Customer-Level RFM + Behavioral Features ---
    snapshot_date = valid_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # First compute per-invoice basket size so AvgBasketSize reflects order volume
    invoice_baskets = (
        valid_df.groupby(["CustomerID", "Invoice"])["Quantity"]
        .sum()
        .reset_index()
    )
    avg_basket_size = invoice_baskets.groupby("CustomerID")["Quantity"].mean()

    rfm = valid_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalSpend", "sum"),
        SkuDiversity=("StockCode", "nunique"),
    )

    rfm["AvgBasketSize"] = avg_basket_size
    rfm["ReturnRate"] = return_rate.reindex(rfm.index).fillna(0.0)

    # Ensure strictly positive values before log1p transformation
    rfm = rfm[rfm["Monetary"] > 0].copy()
    print(f"[Features] Engineered customer profiles: {rfm.shape}")
    return rfm


def build_preprocessing_pipeline(n_components: float = 0.85) -> Pipeline:
    """Executes Phase 3: log1p skewness compression, RobustScaler, and PCA."""
    return Pipeline(
        steps=[
            ("log_transform", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", RobustScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "data" / "raw"
    processed_path = project_root / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    # Run Phases 1 & 2
    raw_df = load_raw_data(raw_path)
    customer_features = clean_and_engineer_features(raw_df)

    # Save clean feature table for EDA and clustering
    output_csv = processed_path / "rfm_features.csv"
    customer_features.to_csv(output_csv)
    print(f"[Saved] Customer feature table written to: {output_csv}")

    # Run Phase 3 & inspect PCA variance retention
    pipeline = build_preprocessing_pipeline(n_components=0.85)
    X_pca = pipeline.fit_transform(customer_features)
    pca_step = pipeline.named_steps["pca"]

    print("\n--- Phase 3: PCA Summary ---")
    print(f"Components retained (>=85% variance): {pca_step.n_components_}")
    for i, var in enumerate(pca_step.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {var:.2%} variance")
    print(
        f"Total Cumulative Variance: {pca_step.explained_variance_ratio_.sum():.2%}"
    )