import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances_argmin
from models import Instagram
from models import Tiktok
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import io
import base64

# --- Instagram Clustering ---
def load_instagram_data():
    data = Instagram.query.with_entities(
        Instagram.post_id,
        Instagram.produk,
        Instagram.post_content,
        Instagram.interaksi,
        Instagram.akun_yang_dijangkau,
        Instagram.view_non_followers
    ).all()

    df = pd.DataFrame(data, columns=["post_id", "produk", "post_content", "interaksi", "jangkauan", "view_non_followers"])
    return df

def cluster_instagram_data(df, n_clusters=3, init_method='k-means++', max_iter=10):
    """
    Menjalankan K-Means clustering pada data Instagram dan menghitung silhouette score.
    Menyimpan riwayat centroid tiap iterasi.
    """
    fitur = df[["interaksi", "jangkauan", "view_non_followers"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fitur)

    # Inisialisasi awal centroid dengan 1 iterasi KMeans
    kmeans = KMeans(n_clusters=n_clusters, init=init_method, n_init=1 if isinstance(init_method, str) else 1, max_iter=300, random_state=42)
    kmeans.fit(X_scaled)
    centers = kmeans.cluster_centers_
    history = [deepcopy(centers)]

    for _ in range(max_iter):
        labels = pairwise_distances_argmin(X_scaled, centers)
        new_centers = np.array([X_scaled[labels == i].mean(axis=0) for i in range(n_clusters)])
        history.append(deepcopy(new_centers))

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    # Final hasil clustering
    df["cluster"] = labels + 1  # mulai dari 1
    df["evaluasi"] = silhouette_samples(X_scaled, labels)
    silhouette_avg = silhouette_score(X_scaled, labels)

    return df, silhouette_avg, history

def kmeans_with_tracking(df, n_clusters=3, max_iter=10, init='k-means++'):
    """
    Melakukan KMeans manual dengan pelacakan centroid, jarak, dan assignment per iterasi.
    Mendukung inisialisasi centroid dengan metode default 'k-means++' atau 'mean' (rata-rata global).
    """
    fitur = df[["interaksi", "jangkauan", "view_non_followers"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fitur)

    if init == 'mean':
        # Hitung rata-rata dari seluruh data sebagai centroid pertama
        mean_vector = np.mean(X_scaled, axis=0)
        centroids = [mean_vector]

        np.random.seed(42)
        # Tambahkan centroid lainnya secara acak dari data
        while len(centroids) < n_clusters:
            random_idx = np.random.choice(len(X_scaled))
            centroids.append(X_scaled[random_idx])

        centers = np.array(centroids)
    else:
        # Gunakan metode default k-means++ dari sklearn
        kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=1, max_iter=1, random_state=42)
        kmeans.fit(X_scaled)
        centers = kmeans.cluster_centers_

    centroid_history = [deepcopy(centers)]
    distance_matrix = []
    assignments = []

    for _ in range(max_iter):
        distances = np.linalg.norm(X_scaled[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        distance_matrix.append(distances.tolist())
        assignments.append(labels.tolist())

        new_centers = np.array([
            X_scaled[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(n_clusters)
        ])

        centroid_history.append(deepcopy(new_centers))

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    return {
        'centroid_history': centroid_history,
        'distance_matrix': distance_matrix,
        'assignments': assignments
    }



def convert_centroid_history_to_df(history):
    hasil = []
    for iterasi, centroid in enumerate(history):
        for i, titik in enumerate(centroid):
            hasil.append({
                "Iterasi": iterasi,
                "Cluster": f"C{i+1}",
                "x": round(titik[0], 4),
                "y": round(titik[1], 4),
                "z": round(titik[2], 4),
            })
    return pd.DataFrame(hasil)

def generate_cluster_plot(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(
        df["interaksi"],
        df["jangkauan"],
        c=df["cluster"],
        cmap="viridis",
        s=50,
        edgecolor='k'
    )
    ax.set_xlabel("Interaksi")
    ax.set_ylabel("Jangkauan")
    ax.set_title("Clustering Instagram (Interaksi vs Jangkauan)")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64

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

def cluster_tiktok_data(df, n_clusters=3):
    fitur = df[["engagement", "tayangan", "view_non_followers"]]
    X_scaled = StandardScaler().fit_transform(fitur)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = model.fit_predict(X_scaled)

    silhouette_vals = silhouette_samples(X_scaled, df["cluster"])
    df["evaluasi"] = silhouette_vals
    silhouette_avg = silhouette_score(X_scaled, df["cluster"] - 1)
    return df, silhouette_avg

def generate_tiktok_plot(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(
        df["engagement"],
        df["tayangan"],
        c=df["cluster"],
        cmap="plasma",
        s=50,
        edgecolor='k'
    )
    ax.set_xlabel("Engagement")
    ax.set_ylabel("Tayangan")
    ax.set_title("Clustering TikTok (Engagement vs Tayangan)")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64

def calculate_silhouette_analysis(df, cluster_labels):
    features = df[["interaksi", "jangkauan", "view_non_followers"]]
    X_scaled = StandardScaler().fit_transform(features)

    avg_score = silhouette_score(X_scaled, cluster_labels)
    sample_scores = silhouette_samples(X_scaled, cluster_labels)

    per_cluster_score = {}
    for cluster in np.unique(cluster_labels):
        per_cluster_score[cluster] = sample_scores[cluster_labels == cluster].mean()

    return avg_score, per_cluster_score, sample_scores

def generate_silhouette_plot(df, cluster_labels):
    features = df[["interaksi", "jangkauan", "view_non_followers"]]
    X_scaled = StandardScaler().fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    silhouette_plot = ax

    sample_scores = silhouette_samples(X_scaled, cluster_labels)
    y_lower = 10

    for i in sorted(np.unique(cluster_labels)):
        ith_cluster_scores = sample_scores[cluster_labels == i]
        ith_cluster_scores.sort()

        size_cluster_i = ith_cluster_scores.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.viridis(float(i) / len(np.unique(cluster_labels)))
        silhouette_plot.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_scores,
                                  facecolor=color, edgecolor=color, alpha=0.7)

        silhouette_plot.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))
        y_lower = y_upper + 10

    silhouette_plot.set_title("Silhouette Plot")
    silhouette_plot.set_xlabel("Silhouette Coefficient")
    silhouette_plot.set_ylabel("Cluster")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return img_base64
