"""
Database models and setup for Courtvision
Uses SQLAlchemy with SQLite for simplicity
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
import os

db = SQLAlchemy()

# JWT secret key (in production, use environment variable)
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'courtvision-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

class User(db.Model):
    """User model for authentication and profile"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    sessions = db.relationship('Session', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if password is correct"""
        return check_password_hash(self.password_hash, password)
    
    def generate_token(self):
        """Generate JWT token for user"""
        payload = {
            'user_id': self.id,
            'username': self.username,
            'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def verify_token(token):
        """Verify JWT token and return user"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            user_id = payload.get('user_id')
            if user_id:
                return User.query.get(user_id)
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        return None
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Session(db.Model):
    """Basketball training session"""
    __tablename__ = 'sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ended_at = db.Column(db.DateTime, nullable=True)
    duration_seconds = db.Column(db.Integer, default=0)  # Duration in seconds
    
    # Session stats
    shots_made = db.Column(db.Integer, default=0)
    shots_missed = db.Column(db.Integer, default=0)
    total_shots = db.Column(db.Integer, default=0)
    accuracy = db.Column(db.Float, default=0.0)  # Percentage
    
    # Relationships
    shots = db.relationship('Shot', backref='session', lazy=True, cascade='all, delete-orphan')
    
    def calculate_accuracy(self):
        """Calculate and update accuracy"""
        if self.total_shots > 0:
            self.accuracy = (self.shots_made / self.total_shots) * 100
        else:
            self.accuracy = 0.0
        return self.accuracy
    
    def end_session(self):
        """End the session and calculate duration"""
        if not self.ended_at:
            self.ended_at = datetime.utcnow()
            if self.started_at:
                delta = self.ended_at - self.started_at
                self.duration_seconds = int(delta.total_seconds())
    
    def to_dict(self):
        """Convert session to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'duration_seconds': self.duration_seconds,
            'shots_made': self.shots_made,
            'shots_missed': self.shots_missed,
            'total_shots': self.total_shots,
            'accuracy': round(self.accuracy, 1)
        }

class Shot(db.Model):
    """Individual shot record"""
    __tablename__ = 'shots'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    made = db.Column(db.Boolean, nullable=False)  # True if made, False if missed
    
    def to_dict(self):
        """Convert shot to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'made': self.made
        }

def init_db(app):
    """Initialize database with Flask app"""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        
        # Create test account if it doesn't exist
        test_user = User.query.filter_by(username='test').first()
        if not test_user:
            test_user = User(
                username='test',
                email='test@courtvision.com'
            )
            test_user.set_password('test123')
            db.session.add(test_user)
            db.session.commit()
            print("✓ Test account created: username='test', password='test123'")
        
        print("✓ Database initialized")
