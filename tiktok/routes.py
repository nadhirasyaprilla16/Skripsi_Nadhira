from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from models import db, Tiktok
from clustering_utils import load_tiktok_data, cluster_tiktok_data, generate_tiktok_plot
import pandas as pd
import math

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

# Clustering TikTok (dengan input jumlah cluster + visualisasi)
@tiktok_bp.route('/clustering', methods=['GET', 'POST'])
def clustering():
    # Ambil data TikTok
    df = load_tiktok_data()

    if df.empty:
        return render_template('tiktok/clustering.html', data=None, page=1, total_pages=1, n_clusters=None, cluster_plot=None)

    # Ambil jumlah cluster dari input form, default = 3
    if request.method == 'POST' :
        n_clusters = int(request.form.get('n_clusters', 3))
        session['n_clusters'] = n_clusters
    else:
        n_clusters = session.get('n_clusters', 3)

    # Proses clustering
    df_clustered, silhoutte_avg = cluster_tiktok_data(df, n_clusters)
    cluster_plot = generate_tiktok_plot(df_clustered)

     # Simpan ke database
    for index, row in df_clustered.iterrows():
        post = Tiktok.query.get(row['post_id'])
        if post:
            post.cluster = int(row['cluster'])
            post.evaluasi = float(row['evaluasi'])

    db.session.commit()

    # Tambahkan kolom evaluasi berdasarkan silhouette score
    df_clustered['evaluasi'] = silhoutte_avg

    # Manual pagination
    page = request.args.get('page', 1, type=int)
    per_page = 5
    total_rows = len(df_clustered)
    total_pages = math.ceil(total_rows / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    current_page_data = df_clustered.iloc[start:end]

    return render_template(
        'tiktok/clustering.html',
        data=current_page_data,
        page=page,
        total_pages=total_pages,
        n_clusters=n_clusters,
        cluster_plot=cluster_plot,
        silhoutte=silhoutte_avg
    )

# Evaluasi TikTok
@tiktok_bp.route('/evaluasi')
def evaluasi():
    return render_template('tiktok/evaluasi.html')
