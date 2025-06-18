from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

# Model Instagram
class Instagram(db.Model):
    __tablename__ = 'instagram'
    post_id = db.Column(db.String(10), primary_key=True)
    produk = db.Column(db.String(100))
    post_content = db.Column(db.Text)
    interaksi = db.Column(db.Integer)
    akun_yang_dijangkau = db.Column(db.Integer)
    view_non_followers = db.Column(db.Float)
    # engagement_rate = db.Column(db.Float)
    post_type = db.Column(db.Integer)
    # cluster = db.Column(db.Integer)
    # evaluasi = db.Column(db.Float)

# Model TikTok
class Tiktok(db.Model):
    __tablename__ = 'tiktok'
    post_id = db.Column(db.String(10), primary_key=True)
    produk = db.Column(db.String(100))
    post_content = db.Column(db.Text)
    engagement = db.Column(db.Integer)
    tayangan = db.Column(db.Integer)
    view_non_followers = db.Column(db.Float)
    # cluster = db.Column(db.Integer)
    # evaluasi = db.Column(db.Float)

class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

