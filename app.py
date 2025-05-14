from flask import Flask, render_template
from models import db
from instagram.routes import instagram_bp
from tiktok.routes import tiktok_bp

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Konfigurasi database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/skripsi'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inisialisasi database
db.init_app(app)

# Register blueprint
app.register_blueprint(instagram_bp, url_prefix='/instagram')
app.register_blueprint(tiktok_bp, url_prefix='/tiktok')

# Tambahkan route beranda
@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
