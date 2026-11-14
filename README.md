# Customer Segmentation & Behavioral Clustering Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

An end-to-end unsupervised Machine Learning pipeline and interactive analytics application that transforms **1,033,036 raw e-commerce transactions** from the **UCI Online Retail II** dataset into mathematically validated customer personas.

Beyond standard 3D RFM (Recency, Frequency, Monetary) scoring, this engine constructs a **6-dimensional behavioral feature space**, applies leak-free logarithmic scaling and Principal Component Analysis (PCA), benchmarks partitioning, probabilistic, and density-based clustering paradigms, and deploys an interactive **Streamlit** dashboard for 3D topology exploration and real-time customer inference.

---

## System Architecture

```text
+-----------------------------+      +----------------------------------+
|  Raw Transactional Logs     |      |  Phase 1: Data Sanitization      |
|  Online Retail II (2009-11) |----->|  - Deduplication (1,033,036)     |
|  Year 2009-2010 & 2010-2011 |      |  - ReturnRate pre-computation    |
+-----------------------------+      |  - Filter cancellations & non-SKU|
                                     +-----------------+----------------+
                                                       |
                                                       v
+-----------------------------+      +----------------------------------+
|  Phase 3: ML Preprocessing  |      |  Phase 2: Behavioral Aggregation |
|  - np.log1p() skew reduction|<-----|  - 5,852 Customer Profiles       |
|  - RobustScaler()           |      |  - RFM + SkuDiversity,           |
|  - PCA (3 PCs = 87.09% var) |      |    AvgBasketSize, ReturnRate     |
+--------------+--------------+      +----------------------------------+
               |
               v
+-----------------------------------------------------------------------+
|  Phase 4 & 5: Unsupervised Benchmarking & Persona Profiling           |
|  - K-Means (k=2 vs. k=4)   - Gaussian Mixture Models (BIC)   - DBSCAN |
|  - Silhouette, Davies-Bouldin, Calinski-Harabasz & t-SNE Validation   |
+----------------------------------+------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|  Interactive Streamlit Application (app.py)                           |
|  - 3D PCA Scatter Explorer - Z-Score Snake Plots - Live Inference     |
+-----------------------------------------------------------------------+
