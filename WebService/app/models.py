from app import db, login
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(60), nullable=False)
    avatar = db.Column(db.String(20), nullable=False, default='avatar.png')
    cover_pic = db.Column(db.String(20), nullable=False, default='cover.png')
    age = db.Column(db.Integer, nullable=False)
    address = db.Column(db.Text, nullable=False)
    register_date = db.Column(
        db.DateTime, nullable=False, default=datetime.utcnow)

    videos = db.relationship('Video', backref='author', lazy='dynamic')
    comments = db.relationship('Comments', backref='author', lazy='dynamic')
    liked = db.relationship(
        'Likes', foreign_keys='Likes.user_id', backref='user', lazy='dynamic')

    def __repr__(self):
        return f"User('{self.user_name}', '{self.email}')"

    def set_password(self, password):
        '''Setting up the password'''

        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        '''Checking the password filled with the password in database'''

        return check_password_hash(self.password_hash, password)

    def like_video(self, video):
        if not self.has_liked_video(video):
            like = Likes(user_id=self.id, video_id=video.id)
            db.session.add(like)

    def unlike_video(self, video):
        if self.has_liked_video(video):
            Likes.query.filter_by(user_id=self.id, video_id=video.id).delete()

    def has_liked_video(self, video):
        return Likes.query.filter(Likes.user_id == self.id, Likes.video_id == video.id).count() > 0


@login.user_loader
def load_user(id):
    '''function to add the user into the session generated by LoginManager'''

    return User.query.get(int(id))


class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_title = db.Column(db.String(120), nullable=False)
    video_content = db.Column(db.String(40), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String, nullable=False)
    views_count = db.Column(db.Integer, nullable=False, default=0)
    upload_time = db.Column(db.DateTime, nullable=False,
                            default=datetime.utcnow)
    likes_count = db.Column(db.Integer, nullable=False, default=0)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    comments = db.relationship('Comments', backref='video', lazy='dynamic')
    likes = db.relationship('Likes', backref='video', lazy='dynamic')

    def __repr__(self):
        return f"Video('{self.video_title}', '{self.upload_time}')"


class Likes(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


class Comments(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.Text)
    comment_time = db.Column(db.DateTime, index=True, default=datetime.utcnow)

    author_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'))

    def __repr__(self):
        return f"Comments('{self.author_id}', '{self.body}', {self.video_id})"