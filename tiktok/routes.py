from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from models import db, Tiktok
from clustering_utils import load_tiktok_data, kmeans_clustering_full_tiktok
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import math
import random
from sklearn.cluster import KMeans

tiktok_bp = Blueprint('tiktok', __name__, template_folder='templates')

# Halaman utama TikTok dengan pagination
@tiktok_bp.route('/')
def index():
    page = request.args.get('page', 1, type=int)
    per_page = 5
    posts = Tiktok.query.paginate(page=page, per_page=per_page)
    return render_template('tiktok/index.html', posts=posts)

# Tambah data TikTok
@tiktok_bp.route('/tambah', methods=['GET', 'POST'])
def tambah():
    if request.method == 'POST':
        post = Tiktok(
            post_id=request.form['post_id'],
            produk=request.form['produk'],
            post_content=request.form['post_content'],
            engagement=int(request.form['engagement']),
            tayangan=int(request.form['tayangan']),
            view_non_followers=float(request.form['view_non_followers'])
        )
        db.session.add(post)
        db.session.commit()
        flash('Data TikTok berhasil ditambahkan', 'success')
        return redirect(url_for('tiktok.index'))
    return render_template('tiktok/tambah.html')

# Route Upload Excel
@tiktok_bp.route('/upload', methods=['GET', 'POST'])
def upload_excel_tiktok():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.xlsx'):
            try:
                df = pd.read_excel(file)

                # Hapus semua data TikTok lama
                Tiktok.query.delete()
                db.session.commit()

                # Tambahkan data baru
                for _, row in df.iterrows():
                    post = Tiktok(
                        post_id=row['Post Id'],
                        produk=row['Produk'],
                        post_content=row['Post Content'],
                        engagement=int(row['Engagement']),
                        tayangan=int(row['Tayangan']),
                        view_non_followers=float(row['views non follower (%)'])
                    )
                    db.session.add(post)
                db.session.commit()

                flash('Data TikTok berhasil diupload dan data lama telah dihapus.', 'success')
                return redirect(url_for('tiktok.index'))  # Redirect ke halaman kelola data TikTok

            except Exception as e:
                flash(f'Terjadi kesalahan saat upload: {e}', 'danger')
                return redirect(url_for('tiktok.index'))

        else:
            flash('File harus berformat .xlsx', 'danger')
            return redirect(url_for('tiktok.index'))

    # GET request tetap tampilkan halaman upload
    return render_template('tiktok/upload.html')


# Edit data TikTok
@tiktok_bp.route('/edit/<string:id>', methods=['GET', 'POST'])
def edit(id):
    post = Tiktok.query.get_or_404(id)
    if request.method == 'POST':
        post.post_id = request.form['post_id']
        post.produk = request.form['produk']
        post.post_content = request.form['post_content']
        post.engagement = int(request.form['engagement'])
        post.tayangan = int(request.form['tayangan'])
        post.view_non_followers = float(request.form['view_non_followers'])
        db.session.commit()
        flash('Data TikTok berhasil diupdate', 'success')
        return redirect(url_for('tiktok.index'))
    return render_template('tiktok/edit.html', post=post)

# Hapus data TikTok
@tiktok_bp.route('/hapus/<string:id>')
def hapus(id):
    post = Tiktok.query.get_or_404(id)
    db.session.delete(post)
    db.session.commit()
    flash('Data TikTok berhasil dihapus', 'success')
    return redirect(url_for('tiktok.index'))



# ------------------- Route Proses Clustering -------------------

@tiktok_bp.route('/proses-clustering', methods=['GET', 'POST'])
def proses_clustering():
    result = None
    n_clusters = 2
    max_iter = 20
    init_method = session.get('init_method', 'mean')
    silhouette = None
    centroid_history_real = []

    if request.method == 'POST':
        n_clusters = int(request.form.get('n_clusters', 2))
        max_iter = int(request.form.get('max_iter', 10))
        init_method = request.form.get('init_method', 'mean')
        manual_centroids = request.form.get('manual_centroids')

        # Load data dan siapkan fitur
        df = load_tiktok_data()
        fitur = df[["engagement", "tayangan", "view_non_followers"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(fitur)

        session['n_clusters'] = n_clusters
        session['max_iter'] = max_iter
        session['init_method'] = init_method

        # Centroid manual jika digunakan
        if init_method == 'manual' and manual_centroids:
            centroid_values = [list(map(float, line.split(','))) for line in manual_centroids.splitlines()]
            centroid_array = np.array(centroid_values)

            mean_vals = fitur.mean().values
            std_vals = fitur.std().values
            centroid_zscore = (centroid_array - mean_vals) / std_vals

            session['manual_init_centroid'] = centroid_zscore.tolist()
            initial_centroids = centroid_zscore
        else:
            session['manual_init_centroid'] = None
            initial_centroids = 'mean'

        # Jalankan clustering
        result = kmeans_clustering_full_tiktok(
            df,
            n_clusters=n_clusters,
            init_method=initial_centroids,
            max_iter=max_iter
        )

        # Simpan hasil penting ke session
        session['centroids'] = np.array(result['centroid_history'][-1]).tolist()
        session['cluster_labels_tiktok'] = result['df']['cluster'].tolist()
        session['silhouette'] = float(result['silhouette'])

        # ✅ Simpan hasil lengkap
        session['cluster_result_tiktok'] = {
            'labels': result['df']['cluster'].tolist(),
            'evaluasi': result['df']['evaluasi'].tolist(),
            'silhouette': float(result['silhouette'])
        }

        silhouette = result['silhouette']
        centroid_history_real = result['centroid_history']

    return render_template(
        'tiktok/proses_clustering.html',  # ⬅️ Pastikan pakai template TikTok
        n_clusters=n_clusters,
        max_iter=max_iter,
        init_method=init_method,
        silhouette=silhouette,
        centroid_history=centroid_history_real,
        distance_matrix=result['distance_matrix'] if result else [],
        assignment_per_iter=result['assignments'] if result else []
    )

# Hasil Clustering
@tiktok_bp.route('/hasil-clustering')
def hasil_clustering():
    df = load_tiktok_data()  # Ambil data TikTok
    n_clusters = session.get('n_clusters', 3)
    cluster_result = session.get('cluster_result_tiktok')

    if not cluster_result:
        flash("Hasil clustering tidak ditemukan. Silakan jalankan proses clustering terlebih dahulu.", "warning")
        return redirect(url_for('tiktok.proses_clustering'))

    # Ambil hasil clustering
    labels = cluster_result.get('labels')
    evaluations = cluster_result.get('evaluasi')
    silhouette = cluster_result.get('silhouette', 0)

    if labels is None or evaluations is None or len(labels) != len(df):
        flash("Data clustering tidak valid atau tidak sesuai.", "danger")
        return redirect(url_for('tiktok.proses_clustering'))

    df['cluster'] = labels
    df['evaluasi'] = evaluations

    # ✅ Hitung rata-rata evaluasi per cluster → urutkan dari tertinggi ke rendah
    cluster_summary = (
        df.groupby("cluster")["evaluasi"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"evaluasi": "avg_evaluasi"})
    ).to_dict(orient="records")

    # ✅ Filter berdasarkan cluster (jika ada)
    cluster_filter = request.args.get('cluster_filter')
    if cluster_filter is not None and cluster_filter != '':
        try:
            df = df[df['cluster'] == int(cluster_filter)]
        except ValueError:
            flash("Filter cluster tidak valid.", "danger")

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
    total_rows = len(df)
    total_pages = (total_rows + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    df_page = df.iloc[start:end]

    return render_template(
        'tiktok/hasil_clustering.html',
        data=df_page,
        n_clusters=n_clusters,
        silhouette=silhouette,
        page=page,
        total_pages=total_pages,
        cluster_filter=cluster_filter,
        cluster_summary=cluster_summary  # ⬅️ dikirim ke template
    )
