from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from models import db, Instagram
from clustering_utils import load_instagram_data, cluster_instagram_data, generate_cluster_plot, calculate_silhouette_analysis, kmeans_with_tracking
import pandas as pd
import math
import numpy as np

instagram_bp = Blueprint('instagram', __name__, template_folder='templates')

# 🏠 Halaman utama Instagram dengan pagination
@instagram_bp.route('/')
def dashboard():
    page = request.args.get('page', 1, type=int)
    per_page = 5
    posts = Instagram.query.paginate(page=page, per_page=per_page)
    return render_template('instagram/index.html', posts=posts)

# ➕ Tambah postingan
@instagram_bp.route('/tambah', methods=['GET', 'POST'])
def tambah_instagram():
    if request.method == 'POST':
        post = Instagram(
            post_id=request.form['post_id'],
            produk=request.form['produk'],
            post_content=request.form['post_content'],
            interaksi=int(request.form['interaksi']),
            akun_yang_dijangkau=int(request.form['akun_yang_dijangkau']),
            view_non_followers=float(request.form['view_non_followers']),
            post_type=int(request.form['post_type'])
        )
        db.session.add(post)
        db.session.commit()
        flash('Data berhasil ditambahkan', 'success')
        return redirect(url_for('instagram.dashboard'))
    return render_template('instagram/tambah.html')

# ✏️ Edit postingan
@instagram_bp.route('/edit/<string:post_id>', methods=['GET', 'POST'])
def edit_instagram(post_id):
    post = Instagram.query.get_or_404(post_id)
    if request.method == 'POST':
        post.produk = request.form['produk']
        post.post_content = request.form['post_content']
        post.interaksi = int(request.form['interaksi'])
        post.akun_yang_dijangkau = int(request.form['akun_yang_dijangkau'])
        post.view_non_followers = float(request.form['view_non_followers'])
        post.post_type = int(request.form['post_type'])
        db.session.commit()
        flash('Data berhasil diupdate', 'success')
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

# 🔄 Proses Clustering (Input dan Perhitungan)
@instagram_bp.route('/proses-clustering', methods=['GET', 'POST'])
def proses_clustering():
    hasil = None
    n_clusters = 3
    max_iter = 10
    init_method = 'k-means++'

    if request.method == 'POST':
        n_clusters = int(request.form.get('n_clusters', 3))
        max_iter = int(request.form.get('max_iter', 10))
        init_method = request.form.get('init_method', 'k-means++')

        session['n_clusters'] = n_clusters
        session['max_iter'] = max_iter
        session['init_method'] = init_method

        df = load_instagram_data()
        hasil = kmeans_with_tracking(df, n_clusters=n_clusters, max_iter=max_iter, init=init_method)

        if init_method == 'mean':
            session['manual_init_centroid'] = hasil['centroid_history'][0].tolist()

    return render_template(
        'instagram/proses_clustering.html',
        n_clusters=n_clusters,
        max_iter=max_iter,
        centroid_history=hasil['centroid_history'] if hasil else [],
        distance_matrix=hasil['distance_matrix'] if hasil else [],
        assignment_per_iter=hasil['assignments'] if hasil else []
    )

# 📊 Hasil Clustering (Output akhir dan evaluasi)
@instagram_bp.route('/hasil-clustering')
def hasil_clustering():
    df = load_instagram_data()
    n_clusters = session.get('n_clusters', 3)
    init_method = session.get('init_method', 'k-means++')

    if init_method == 'mean' and 'manual_init_centroid' in session:
        init_value = np.array(session['manual_init_centroid'])
    else:
        init_value = init_method

    df_clustered, silhouette_score, _ = cluster_instagram_data(
        df,
        n_clusters=n_clusters,
        init_method=init_value
    )

    for _, row in df_clustered.iterrows():
        post = Instagram.query.get(row['post_id'])
        if post:
            post.cluster = int(row['cluster'])
            post.evaluasi = float(row['evaluasi'])
    db.session.commit()

    return render_template(
        'instagram/hasil_clustering.html',
        data=df_clustered,
        silhouette=silhouette_score,
        n_clusters=n_clusters
    )

# ✅ Evaluasi clustering
@instagram_bp.route('/evaluasi')
def evaluasi():
    df = load_instagram_data()

    if df.empty:
        flash("Belum ada data untuk dievaluasi", "warning")
        return redirect(url_for('instagram.dashboard'))

    avg_score, per_cluster_score, _ = calculate_silhouette_analysis(df, df['cluster'])

    return render_template(
        'instagram/evaluasi.html',
        avg_score=avg_score,
        cluster_scores=per_cluster_score,
        n_clusters=len(per_cluster_score)
    )
