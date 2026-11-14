# Customer Segmentation & Behavioral Clustering Engine

An end-to-end unsupervised Machine Learning pipeline that transforms ~1.03M raw e-commerce transactions from the **Online Retail II** dataset into actionable customer personas using RFM + behavioral feature engineering, PCA dimensionality reduction, multi-algorithm clustering benchmarks, and an interactive Streamlit dashboard.

## Key Results
- **Data Sanitization:** Filtered 1,033,036 deduplicated transactions down to **776,579 valid merchandise purchases** across **5,852 unique customers**.
- **Dimensionality Reduction (PCA):** 3 principal components capture **87.09%** of cumulative behavioral variance (PC1: 53.84%, PC2: 18.18%, PC3: 15.07%).
- **Algorithm Benchmark:**
  - **K-Means (k=2):** Silhouette = 0.3474 | Davies-Bouldin = 1.1052 | Calinski-Harabasz = 3816.00
  - **GMM (k=9, full):** Silhouette = 0.1116 | Davies-Bouldin = 1.5279 | Calinski-Harabasz = 1051.59
  - **DBSCAN (eps=0.43):** Isolates 2.65% wholesale outliers as noise on continuous RFM density.
- **Business Personas (K-Means, k=4):**
  - **VIP Champions & Wholesale Loyalists (27.8%):** Median Recency 22d, Frequency 11, Spend £3,921.10
  - **Promising Occasional Bulk Buyers (33.4%):** Median Recency 157d, Frequency 2, Spend £623.28, 0% Return Rate
  - **High-Return / At-Risk Mid-Tier Buyers (19.2%):** Median Recency 188d, Frequency 3, Spend £945.04, 33% Return Rate
  - **Dormant Low-Value One-Timers (19.6%):** Median Recency 366d, Frequency 1, Spend £174.80

## How to Run Locally
1. Create and activate virtual environment: python -m venv .venv && .\.venv\Scriptsctivate
2. Install dependencies: pip install -r requirements.txt
3. Run preprocessing: python src/preprocessing.py
4. Run clustering evaluation: python src/clustering.py
5. Launch dashboard: streamlit run app.py
