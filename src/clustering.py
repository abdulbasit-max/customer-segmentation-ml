from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from preprocessing import build_preprocessing_pipeline


def find_optimal_kmeans(X_pca: np.ndarray, k_range=range(2, 11), out_dir: Path = None) -> int:
    """Sweeps k to compute WCSS (Inertia) and Silhouette scores, saving an Elbow + Silhouette plot."""
    wcss = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
        labels = km.fit_predict(X_pca)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X_pca, labels))

    best_k = list(k_range)[int(np.argmax(sil_scores))]

    if out_dir:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(list(k_range), wcss, marker="o", color="#1f77b4", linewidth=2)
        axes[0].set_title("K-Means Elbow Curve (WCSS)")
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].set_ylabel("Inertia (WCSS)")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].plot(list(k_range), sil_scores, marker="s", color="#2ca02c", linewidth=2)
        axes[1].axvline(best_k, color="red", linestyle="--", label=f"Best k = {best_k}")
        axes[1].set_title("K-Means Silhouette Analysis")
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Silhouette Coefficient")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(out_dir / "kmeans_selection.png", dpi=300)
        plt.close()

    print(f"[K-Means] Optimal k by Silhouette Score: k={best_k} (Score={max(sil_scores):.4f})")
    return best_k


def find_optimal_gmm(X_pca: np.ndarray, k_range=range(2, 11), out_dir: Path = None):
    """Selects optimal GMM component count and covariance type using lowest BIC."""
    cov_types = ["full", "tied", "diag"]
    bic_records = []

    best_bic = np.inf
    best_params = None

    for cov in cov_types:
        for k in k_range:
            gmm = GaussianMixture(
                n_components=k, covariance_type=cov, n_init=10, random_state=42
            )
            gmm.fit(X_pca)
            bic = gmm.bic(X_pca)
            bic_records.append({"k": k, "covariance_type": cov, "BIC": bic})
            if bic < best_bic:
                best_bic = bic
                best_params = (k, cov)

    bic_df = pd.DataFrame(bic_records)
    if out_dir:
        plt.figure(figsize=(8, 4.5))
        sns.lineplot(data=bic_df, x="k", y="BIC", hue="covariance_type", marker="o")
        plt.title("GMM Model Selection via Bayesian Information Criterion (BIC)")
        plt.xlabel("Number of Components (k)")
        plt.ylabel("BIC (Lower is Better)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_dir / "gmm_bic_selection.png", dpi=300)
        plt.close()

    print(f"[GMM] Best configuration by BIC: k={best_params[0]}, covariance='{best_params[1]}'")
    return best_params


def evaluate_models(X_pca: np.ndarray, out_dir: Path):
    """Trains K-Means, GMM, and DBSCAN and benchmarks internal validation metrics."""
    # 1. K-Means
    best_k = find_optimal_kmeans(X_pca, out_dir=out_dir)
    kmeans = KMeans(n_clusters=best_k, init="k-means++", n_init=20, random_state=42)
    km_labels = kmeans.fit_predict(X_pca)

    # 2. Gaussian Mixture Model
    gmm_k, gmm_cov = find_optimal_gmm(X_pca, out_dir=out_dir)
    gmm = GaussianMixture(
        n_components=gmm_k, covariance_type=gmm_cov, n_init=10, random_state=42
    )
    gmm_labels = gmm.fit_predict(X_pca)

    # 3. DBSCAN (tuned via 95th percentile k-distance heuristic)
    min_samples = 10
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X_pca)
    distances, _ = nn.kneighbors(X_pca)
    k_distances = np.sort(distances[:, -1])
    eps_estimate = float(np.percentile(k_distances, 95))

    dbscan = DBSCAN(eps=eps_estimate, min_samples=min_samples)
    db_labels = dbscan.fit_predict(X_pca)

    # Compute validation metrics (for DBSCAN, evaluate on non-noise points)
    results = []
    for name, labels in [
        (f"K-Means (k={best_k})", km_labels),
        (f"GMM (k={gmm_k}, {gmm_cov})", gmm_labels),
        (f"DBSCAN (eps={eps_estimate:.2f})", db_labels),
    ]:
        mask = labels != -1
        unique_clusters = len(set(labels[mask]))
        noise_pct = (1.0 - mask.mean()) * 100

        if unique_clusters >= 2:
            sil = silhouette_score(X_pca[mask], labels[mask])
            dbi = davies_bouldin_score(X_pca[mask], labels[mask])
            chs = calinski_harabasz_score(X_pca[mask], labels[mask])
        else:
            sil, dbi, chs = np.nan, np.nan, np.nan

        results.append(
            {
                "Algorithm": name,
                "Clusters Found": unique_clusters,
                "Noise (%)": round(noise_pct, 2),
                "Silhouette (Higher=Better)": round(sil, 4),
                "Davies-Bouldin (Lower=Better)": round(dbi, 4),
                "Calinski-Harabasz (Higher=Better)": round(chs, 2),
            }
        )

    benchmark_df = pd.DataFrame(results)
    return benchmark_df, km_labels, gmm_labels, db_labels


def plot_tsne_and_snake(
    features_df: pd.DataFrame, X_pca: np.ndarray, labels: np.ndarray, out_dir: Path
):
    """Generates 2D t-SNE cluster topology and normalized snake plot for persona profiling."""
    # 2D t-SNE Projection
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, init="pca")
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=labels,
        palette="tab10",
        s=25,
        alpha=0.8,
    )
    plt.title("2D t-SNE Manifold Projection of Customer Segments (K-Means)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Cluster ID")
    plt.tight_layout()
    plt.savefig(out_dir / "tsne_segments.png", dpi=300)
    plt.close()

    # Standardized Snake Plot for Business Persona Interpretation
    norm_df = pd.DataFrame(
        (np.log1p(features_df) - np.log1p(features_df).mean())
        / np.log1p(features_df).std(),
        columns=features_df.columns,
        index=features_df.index,
    )
    norm_df["Cluster"] = labels
    melted = norm_df.melt(
        id_vars="Cluster", var_name="Behavioral_Feature", value_name="Z_Score"
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=melted,
        x="Behavioral_Feature",
        y="Z_Score",
        hue="Cluster",
        palette="tab10",
        marker="o",
        linewidth=2.2,
    )
    plt.axhline(0, color="gray", linestyle="--", alpha=0.6)
    plt.title("Customer Persona Snake Plot (Standardized Log-Feature Profiles)")
    plt.ylabel("Standardized Value (Z-Score)")
    plt.xlabel("Behavioral Metric")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_dir / "persona_snake_plot.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    processed_path = project_root / "data" / "processed"
    reports_dir = project_root / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load engineered RFM features
    rfm_df = pd.read_csv(processed_path / "rfm_features.csv", index_col="CustomerID")

    # Apply Phase 3 preprocessing pipeline
    pipeline = build_preprocessing_pipeline(n_components=0.85)
    X_pca = pipeline.fit_transform(rfm_df)

    # Run Phase 4 & 5 model comparison
    benchmark_df, km_labels, gmm_labels, db_labels = evaluate_models(X_pca, reports_dir)

    print("\n=== Phase 5: Clustering Benchmark Comparison ===")
    print(benchmark_df.to_string(index=False))
    benchmark_df.to_csv(processed_path / "model_benchmark.csv", index=False)

    # Attach cluster assignments & summarize medians in original business units
    rfm_df["KMeans_Cluster"] = km_labels
    rfm_df["GMM_Cluster"] = gmm_labels
    rfm_df["DBSCAN_Cluster"] = db_labels
    rfm_df.to_csv(processed_path / "segmented_customers.csv")

    print("\n=== K-Means Cluster Profiles (Median Original Units) ===")
    profile_summary = rfm_df.groupby("KMeans_Cluster")[
        ["Recency", "Frequency", "Monetary", "SkuDiversity", "AvgBasketSize", "ReturnRate"]
    ].median()
    profile_summary["Customer_Count"] = rfm_df.groupby("KMeans_Cluster").size()
    print(profile_summary.round(2).to_string())
    profile_summary.to_csv(processed_path / "cluster_profiles.csv")

    # Generate t-SNE & Snake plots
    plot_tsne_and_snake(
        rfm_df[["Recency", "Frequency", "Monetary", "SkuDiversity", "AvgBasketSize", "ReturnRate"]],
        X_pca,
        km_labels,
        reports_dir,
    )
    print(f"\n[Saved] All evaluation figures written to: {reports_dir}")


    # Also fit k=4 K-Means (Elbow inflection point for granular business personas)
    km4 = KMeans(n_clusters=4, init="k-means++", n_init=20, random_state=42)
    rfm_df["KMeans_k4_Cluster"] = km4.fit_predict(X_pca)
    rfm_df.to_csv(processed_path / "segmented_customers.csv")

    print("\n=== K-Means (k=4) Granular Business Profiles (Median Original Units) ===")
    k4_summary = rfm_df.groupby("KMeans_k4_Cluster")[
        ["Recency", "Frequency", "Monetary", "SkuDiversity", "AvgBasketSize", "ReturnRate"]
    ].median()
    k4_summary["Customer_Count"] = rfm_df.groupby("KMeans_k4_Cluster").size()
    print(k4_summary.round(2).to_string())
    k4_summary.to_csv(processed_path / "cluster_profiles_k4.csv")