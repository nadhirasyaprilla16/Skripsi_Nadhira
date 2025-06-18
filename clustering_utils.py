import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_distances
from models import Instagram
from models import Tiktok
import numpy as np
from copy import deepcopy
history = []
import matplotlib.pyplot as plt
import io
import base64

# --- Instagram Clustering ---
# Persiapan Data
def load_instagram_data():
    data = Instagram.query.with_entities(
        Instagram.post_id,
        Instagram.produk,
        Instagram.post_content,
        Instagram.interaksi,
        Instagram.akun_yang_dijangkau,
        Instagram.view_non_followers,
        Instagram.post_type
    ).all()

    df = pd.DataFrame(data, columns=["post_id", "produk", "post_content", "interaksi", "jangkauan", "view_non_followers", "post_type"])
    return df

def kmeans_clustering_full(df, n_clusters=3, init_method='mean', max_iter=100):
    # Step 1: Standardisasi data
    fitur = df[["interaksi", "jangkauan", "view_non_followers"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fitur)

    # Step 2: Inisialisasi centroid
    if isinstance(init_method, str):
        if init_method == 'mean':
            mean_vector = np.mean(X_scaled, axis=0)
            centroids = [mean_vector]
            np.random.seed(42)
            while len(centroids) < n_clusters:
                idx = np.random.choice(len(X_scaled))
                centroids.append(X_scaled[idx])
            centers = np.array(centroids)

        elif init_method == 'kmeans++':
            model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, max_iter=1)
            model.fit(X_scaled)
            centers = model.cluster_centers_

        else:
            raise ValueError("Metode centroid tidak dikenali. Gunakan 'mean', 'kmeans++', atau centroid manual (np.ndarray).")

    elif isinstance(init_method, np.ndarray):
        init_method = scaler.transform(init_method)
        if init_method.shape[1] != X_scaled.shape[1]:
            raise ValueError("Dimensi centroid manual tidak cocok.")
        centers = init_method

    else:
        raise ValueError("Format centroid tidak valid.")

    # Step 3: Iterasi manual
    centroid_history = [deepcopy(centers)]
    distance_matrix = []
    assignments = []

    for _ in range(max_iter):
        distances = np.linalg.norm(X_scaled[:, np.newaxis] - centers, axis=2)
        distance_matrix.append(distances.tolist())

        labels = np.argmin(distances, axis=1)
        assignments.append(labels.tolist())

        new_centers = np.array([
            X_scaled[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(n_clusters)
        ])

        centroid_history.append(deepcopy(new_centers))

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    # Step 4: Evaluasi
    if len(np.unique(labels)) < 2:
        silhouette_avg = 0
        sample_silhouette = [0] * len(X_scaled)
    else:
        sample_silhouette = silhouette_samples(X_scaled, labels)
        silhouette_avg = silhouette_score(X_scaled, labels)

    # Step 5: Gabungkan hasil ke DataFrame baru
    df_result = df.copy()
    df_result["cluster"] = labels
    df_result["evaluasi"] = sample_silhouette

    # Step 6: Return lengkap
    return {
        "df": df_result,
        "silhouette": silhouette_avg,
        "centroid_history": centroid_history,
        "assignments": assignments,
        "distance_matrix": distance_matrix,
        "labels": labels,
        "X_scaled": X_scaled,
    }





def load_tiktok_data():
    data = Tiktok.query.with_entities(
        Tiktok.post_id,
        Tiktok.produk,
        Tiktok.post_content,
        Tiktok.engagement,
        Tiktok.tayangan,
        Tiktok.view_non_followers
    ).all()

    return pd.DataFrame(data, columns=["post_id", "produk", "post_content", "engagement", "tayangan", "view_non_followers"])


def kmeans_clustering_full_tiktok(df, n_clusters, init_method='mean', max_iter=100):
    # Step 1: Persiapan data (standardisasi)
    fitur = df[["engagement", "tayangan", "view_non_followers"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fitur)

    # Step 2: Inisialisasi centroid
    if isinstance(init_method, str) and init_method == 'mean':
        mean_vector = np.mean(X_scaled, axis=0)
        centroids = [mean_vector]
        np.random.seed(42)
        while len(centroids) < n_clusters:
            idx = np.random.choice(len(X_scaled))
            centroids.append(X_scaled[idx])
        centers = np.array(centroids)

    elif isinstance(init_method, np.ndarray):
        init_method = scaler.transform(init_method)
        if init_method.shape[1] != X_scaled.shape[1]:
            raise ValueError("Dimensi centroid manual tidak cocok.")
        centers = init_method

    else:
        raise ValueError("Metode inisialisasi centroid tidak dikenali. Gunakan 'mean' atau centroid manual.")

    # Step 3: Iterasi manual
    centroid_history = [deepcopy(centers)]
    distance_matrix = []
    assignments = []

    for _ in range(max_iter):
        distances = np.linalg.norm(X_scaled[:, np.newaxis] - centers, axis=2)
        distance_matrix.append(distances.tolist())

        labels = np.argmin(distances, axis=1)
        assignments.append(labels.tolist())

        new_centers = np.array([
            X_scaled[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(n_clusters)
        ])

        centroid_history.append(deepcopy(new_centers))

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    # Step 4: Evaluasi
    if len(np.unique(labels)) < 2:
        silhouette_avg = 0
        sample_silhouette = [0] * len(X_scaled)
    else:
        sample_silhouette = silhouette_samples(X_scaled, labels)
        silhouette_avg = silhouette_score(X_scaled, labels)

    # Step 5: Hasil dataframe baru
    df_result = df.copy()
    df_result["cluster"] = labels
    df_result["evaluasi"] = sample_silhouette

    # Step 6: Return
    return {
        "df": df_result,
        "silhouette": silhouette_avg,
        "centroid_history": centroid_history,
        "assignments": assignments,
        "distance_matrix": distance_matrix,
        "labels": labels,
        "X_scaled": X_scaled
    }



# 🔎 Fungsi Generate Scatter Plot Cluster
def generate_cluster_plot(platform, df, labels):
    if labels is None:
        return
    df['cluster'] = labels

    if platform == 'instagram':
        x_col = 'interaksi'
        y_col = 'jangkauan'
    elif platform == 'tiktok':
        x_col = 'engagement'
        y_col = 'tayangan'
    else:
        raise ValueError("Platform tidak dikenal.")

    plt.figure(figsize=(6,4))
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(
            cluster_data[x_col],
            cluster_data[y_col],
            label=f'Cluster {cluster+1}',
            alpha=0.6
        )
    plt.xlabel(x_col.capitalize())
    plt.ylabel(y_col.capitalize())
    plt.title(f'Visualisasi Cluster {platform.capitalize()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'static/visual_{platform}.png')
    plt.close()