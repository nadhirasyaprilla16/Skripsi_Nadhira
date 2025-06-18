from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from models import db, Instagram
from clustering_utils import load_instagram_data, kmeans_clustering_full
import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

instagram_bp = Blueprint('instagram', __name__, template_folder='templates')

# 🏠 Halaman utama Instagram dengan pagination
@instagram_bp.route('/')
def dashboard():
    page = request.args.get('page', 1, type=int)
    per_page = 5
    posts = Instagram.query.paginate(page=page, per_page=per_page)
    return render_template('instagram/index.html', posts=posts)

# index
@instagram_bp.route('/')
def index():
    posts = Instagram.query.all()
    return render_template('instagram/index.html', posts=posts)

# ➕ Tambah postingan
@instagram_bp.route('/tambah', methods=['GET', 'POST'])
def tambah_instagram():
    if request.method == 'POST':
        # Ambil data dan strip spasi
        post_id = request.form.get('post_id', '').strip()
        produk = request.form.get('produk', '').strip()
        post_content = request.form.get('post_content', '').strip()
        interaksi_str = request.form.get('interaksi', '').strip()
        jangkauan_str = request.form.get('akun_yang_dijangkau', '').strip()
        view_str = request.form.get('view_non_followers', '').strip()
        post_type_str = request.form.get('post_type', '').strip()

        # Validasi input kosong
        if not all([post_id, produk, post_content, interaksi_str, jangkauan_str, view_str, post_type_str]):
            flash('Semua field wajib diisi.', 'danger')
            return redirect(request.url)

        # Validasi konversi angka
        try:
            interaksi = int(interaksi_str)
            jangkauan = int(jangkauan_str)
            view_non_followers = float(view_str)
            post_type = int(post_type_str)
        except ValueError:
            flash('Interaksi, Akun yang Dijangkau, View, dan Post Type harus berupa angka.', 'danger')
            return redirect(request.url)

        # Simpan ke database jika semua valid
        post = Instagram(
            post_id=post_id,
            produk=produk,
            post_content=post_content,
            interaksi=interaksi,
            akun_yang_dijangkau=jangkauan,
            view_non_followers=view_non_followers,
            post_type=post_type
        )
        db.session.add(post)
        db.session.commit()
        flash('Data berhasil ditambahkan.', 'success')
        return redirect(url_for('instagram.dashboard'))

    return render_template('instagram/tambah.html')

# Route upload file Excel
@instagram_bp.route('/upload', methods=['GET', 'POST'])
def upload_excel():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.xlsx'):
            try:
                df = pd.read_excel(file)

                # Hapus semua data lama
                Instagram.query.delete()
                db.session.commit()

                # Tambahkan data baru
                for _, row in df.iterrows():
                    post = Instagram(
                        post_id=row['Post Id'],
                        produk=row['Produk'],
                        post_content=row['Post Content'],
                        interaksi=int(row['Interaksi']),
                        akun_yang_dijangkau=int(row['akun yg dijangkau']),
                        view_non_followers=float(row['view non follower (%)']),
                        post_type=int(row['Post Type'])
                    )
                    db.session.add(post)
                db.session.commit()

                # Flash message berhasil
                flash('Data berhasil diupload dan diganti.', 'success')
                return redirect(url_for('instagram.index'))  # Langsung ke kelola data

            except Exception as e:
                # Flash message error
                flash(f'Terjadi kesalahan saat upload: {e}', 'danger')
                return redirect(url_for('instagram.index'))  # Tetap redirect ke kelola data

        else:
            flash('File harus berformat .xlsx', 'danger')
            return redirect(url_for('instagram.index'))  # Tetap redirect ke kelola data

    # GET request tetap tampilkan halaman upload
    return render_template('instagram/upload.html')

# ✏️ Edit postingan
@instagram_bp.route('/edit/<string:post_id>', methods=['GET', 'POST'])
def edit_instagram(post_id):
    post = Instagram.query.get_or_404(post_id)

    if request.method == 'POST':
        # Ambil dan strip semua nilai
        produk = request.form.get('produk', '').strip()
        post_content = request.form.get('post_content', '').strip()
        interaksi_str = request.form.get('interaksi', '').strip()
        jangkauan_str = request.form.get('akun_yang_dijangkau', '').strip()
        view_str = request.form.get('view_non_followers', '').strip()
        post_type_str = request.form.get('post_type', '').strip()

        # Cek jika ada yang kosong
        if not all([produk, post_content, interaksi_str, jangkauan_str, view_str, post_type_str]):
            flash('Semua field wajib diisi.', 'danger')
            return render_template('instagram/edit.html', post=post)

        # Cek konversi angka
        try:
            post.interaksi = int(interaksi_str)
            post.akun_yang_dijangkau = int(jangkauan_str)
            post.view_non_followers = float(view_str)
            post.post_type = int(post_type_str)
        except ValueError:
            flash('Interaksi, Jangkauan, View, dan Post Type harus berupa angka.', 'danger')
            return render_template('instagram/edit.html', post=post)

        # Update field lain
        post.produk = produk
        post.post_content = post_content

        db.session.commit()
        flash('Data berhasil diupdate.', 'success')
        return redirect(url_for('instagram.dashboard'))

    return render_template('instagram/edit.html', post=post)


# ❌ Hapus postingan
@instagram_bp.route('/hapus/<string:post_id>')
def hapus_instagram(post_id):
    post = Instagram.query.get_or_404(post_id)
    db.session.delete(post)
    db.session.commit()
    flash('Data berhasil dihapus', 'success')
    return redirect(url_for('instagram.dashboard'))

# 📂 Halaman kelola data
@instagram_bp.route('/kelola')
def kelola_data():
    page = request.args.get('page', 1, type=int)
    per_page = 5
    posts = Instagram.query.paginate(page=page, per_page=per_page)
    return render_template('instagram/kelola.html', posts=posts)

# Proses Clustering
@instagram_bp.route('/proses-clustering', methods=['GET', 'POST'])
def proses_clustering():
    result = None
    n_clusters = 3
    max_iter = 20
    init_method = session.get('init_method', 'mean')
    silhouette = None
    centroid_history_real = []

    if request.method == 'POST':
        n_clusters = int(request.form.get('n_clusters', 2))
        max_iter = int(request.form.get('max_iter', 10))
        init_method = request.form.get('init_method', 'mean')
        manual_centroids = request.form.get('manual_centroids')

        # Load dan siapkan data
        df = load_instagram_data()
        fitur = df[["interaksi", "jangkauan", "view_non_followers"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(fitur)

        session['n_clusters'] = n_clusters
        session['max_iter'] = max_iter
        session['init_method'] = init_method

        # 🎯 Tentukan initial_centroids sesuai metode
        if init_method == 'manual' and manual_centroids:
            # Centroid manual diubah ke z-score
            centroid_values = [list(map(float, line.split(','))) for line in manual_centroids.splitlines()]
            centroid_array = np.array(centroid_values)

            mean_vals = fitur.mean().values
            std_vals = fitur.std().values
            centroid_zscore = (centroid_array - mean_vals) / std_vals

            session['manual_init_centroid'] = centroid_zscore.tolist()
            initial_centroids = centroid_zscore

        elif init_method == 'mean':
            session['manual_init_centroid'] = None
            initial_centroids = 'mean'

        elif init_method == 'kmeans++':
            session['manual_init_centroid'] = None
            initial_centroids = 'kmeans++'

        else:
            flash("Metode inisialisasi centroid tidak dikenali.", "danger")
            return redirect(url_for('instagram.proses_clustering'))

        # Jalankan clustering
        result = kmeans_clustering_full(df, n_clusters=n_clusters, init_method=initial_centroids, max_iter=max_iter)

        # Simpan hasil ke session
        session['centroids'] = np.array(result['centroid_history'][-1]).tolist()
        session['cluster_labels_instagram'] = result['df']['cluster'].tolist()
        session['silhouette'] = float(result['silhouette'])

        # ✅ Simpan hasil lengkap (digunakan untuk halaman hasil dan evaluasi)
        session['cluster_result_instagram'] = {
            'labels': result['df']['cluster'].tolist(),
            'evaluasi': result['df']['evaluasi'].tolist(),
            'silhouette': float(result['silhouette'])
        }

        silhouette = result['silhouette']
        centroid_history_real = result['centroid_history']

    return render_template(
        'instagram/proses_clustering.html',
        n_clusters=n_clusters,
        max_iter=max_iter,
        init_method=init_method,
        silhouette=silhouette,
        centroid_history=centroid_history_real,
        distance_matrix=result['distance_matrix'] if result else [],
        assignment_per_iter=result['assignments'] if result else []
    )


# 📊 Hasil Clustering (Output akhir dan evaluasi)
@instagram_bp.route('/hasil-clustering')
def hasil_clustering():
    df = load_instagram_data()
    n_clusters = session.get('n_clusters', 3)
    cluster_result = session.get('cluster_result_instagram')

    if not cluster_result:
        flash("Hasil clustering tidak ditemukan. Silakan jalankan proses clustering terlebih dahulu.", "warning")
        return redirect(url_for('instagram.proses_clustering'))

    labels = cluster_result.get('labels')
    evaluations = cluster_result.get('evaluasi')
    silhouette = cluster_result.get('silhouette', 0)

    if labels is None or evaluations is None or len(labels) != len(df):
        flash("Data clustering tidak lengkap atau tidak valid.", "danger")
        return redirect(url_for('instagram.proses_clustering'))

    df['cluster'] = labels
    df['evaluasi'] = evaluations

    # ✅ Hitung rata-rata evaluasi tiap cluster untuk peringkat keterlibatan audiens
    cluster_summary = (
        df.groupby("cluster")["evaluasi"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"evaluasi": "avg_evaluasi"})
    ).to_dict(orient="records")

    # ✅ Ambil parameter filter cluster
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
        'instagram/hasil_clustering.html',
        data=df_page,
        n_clusters=n_clusters,
        silhouette=silhouette,
        page=page,
        total_pages=total_pages,
        cluster_filter=cluster_filter,
        cluster_summary=cluster_summary  # ⬅️ dikirim ke template untuk ditampilkan
    )
