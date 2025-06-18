from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
from models import User, db
from instagram.routes import instagram_bp
from tiktok.routes import tiktok_bp
from clustering_utils import (load_instagram_data, load_tiktok_data, generate_cluster_plot)



app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Konfigurasi database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/skripsi'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inisialisasi database
db.init_app(app)

# Inisialisasi Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Register blueprint
app.register_blueprint(instagram_bp, url_prefix='/instagram')
app.register_blueprint(tiktok_bp, url_prefix='/tiktok')

# Tambahkan route beranda
@app.route('/')
@login_required
def dashboard():
    # Instagram Summary
    df_instagram = load_instagram_data()
    total_instagram = len(df_instagram)
    labels_instagram = session.get('cluster_labels_instagram')
    dominant_cluster_instagram = "-"
    if labels_instagram is not None:
        dominant_cluster_instagram = max(set(labels_instagram), key=labels_instagram.count) + 1
        generate_cluster_plot('instagram', df_instagram, labels_instagram)

    # TikTok Summary
    df_tiktok = load_tiktok_data()
    total_tiktok = len(df_tiktok)
    labels_tiktok = session.get('cluster_labels_tiktok')
    dominant_cluster_tiktok = "-"
    if labels_tiktok is not None:
        dominant_cluster_tiktok = max(set(labels_tiktok), key=labels_tiktok.count) + 1
        generate_cluster_plot('tiktok', df_tiktok, labels_tiktok)

    return render_template(
        'home.html',
        total_instagram=total_instagram,
        total_tiktok=total_tiktok,
        dominant_cluster_instagram=dominant_cluster_instagram,
        dominant_cluster_tiktok=dominant_cluster_tiktok
    )


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    user = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.password == password:  # Pastikan di-deploy pakai password hash ya
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Username atau password salah.')

    return render_template('login.html')

# Loader user untuk Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
