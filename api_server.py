"""
Flask API server for Courtvision webapp
Connects the React frontend with the Python basketball tracking system
"""
from flask import Flask, jsonify, Response, request
from flask_cors import CORS
import cv2
import threading
import time
import base64
import numpy as np
from functools import wraps
from datetime import datetime, timedelta
from live_camera_detection import LiveCameraDetection
from database import db, User, Session, Shot, init_db, JWT_SECRET_KEY, JWT_ALGORITHM
import jwt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///courtvision.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)  # Enable CORS for React frontend

# Initialize database
init_db(app)

# Global tracking system instance
tracker = None
tracker_thread = None
camera_active = False
current_session = None  # Current active session
current_user = None  # Current logged in user
current_stats = {
    'score': 0,
    'misses': 0,
    'total_shots': 0,
    'accuracy': 0.0,
    'players_detected': 0,
    'camera_active': False
}

def get_token_from_header():
    """Extract JWT token from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header.split(' ')[1]
    return None

def token_required(f):
    """Decorator to require JWT authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_header()
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            user_id = payload.get('user_id')
            user = User.query.get(user_id)
            if not user:
                return jsonify({'error': 'Invalid token'}), 401
            request.current_user = user
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated

def init_tracker():
    """Initialize the basketball tracking system"""
    global tracker
    try:
        tracker = LiveCameraDetection('best.pt')
        print("✓ Tracking system initialized")
        return True
    except Exception as e:
        print(f"✗ Error initializing tracker: {e}")
        return False

def update_stats(save_to_db=True):
    """Update current stats from tracker and save to session"""
    global current_stats, tracker, current_session
    if tracker:
        total_shots = tracker.score + tracker.misses
        accuracy = (tracker.score / total_shots * 100) if total_shots > 0 else 0.0
        
        # Update current session in database (only if we have application context)
        if save_to_db and current_session:
            try:
                from flask import has_app_context
                if has_app_context():
                    current_session.shots_made = tracker.score
                    current_session.shots_missed = tracker.misses
                    current_session.total_shots = total_shots
                    current_session.calculate_accuracy()
                    db.session.commit()
            except Exception as e:
                # Silently fail if we can't commit (e.g., outside app context)
                pass
        
        current_stats = {
            'score': tracker.score,
            'misses': tracker.misses,
            'total_shots': total_shots,
            'accuracy': round(accuracy, 1),
            'players_detected': 0,  # Player tracking not implemented in LiveCameraDetection yet
            'camera_active': camera_active
        }

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'API is running'})

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Username, email, and password are required'}), 400
    
    # Check if user already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    # Create new user
    user = User(
        username=data['username'],
        email=data['email']
    )
    user.set_password(data['password'])
    
    try:
        db.session.add(user)
        db.session.commit()
        token = user.generate_token()
        return jsonify({
            'status': 'success',
            'message': 'User registered successfully',
            'token': token,
            'user': user.to_dict()
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user - supports both username and email"""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Username/email and password are required'}), 400
    
    # Try to find user by username or email
    identifier = data['username']
    user = User.query.filter(
        (User.username == identifier) | (User.email == identifier)
    ).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid username/email or password'}), 401
    
    token = user.generate_token()
    return jsonify({
        'status': 'success',
        'message': 'Login successful',
        'token': token,
        'user': user.to_dict()
    })

@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_current_user():
    """Get current authenticated user"""
    return jsonify({
        'status': 'success',
        'user': request.current_user.to_dict()
    })

@app.route('/api/stats', methods=['GET'])
@token_required
def get_stats():
    """Get current tracking statistics and all-time stats"""
    user = request.current_user
    update_stats()
    
    # Get all-time stats from database
    all_sessions = Session.query.filter_by(user_id=user.id).all()
    total_sessions = len(all_sessions)
    total_shots_made = sum(s.shots_made for s in all_sessions)
    total_shots_missed = sum(s.shots_missed for s in all_sessions)
    total_shots = total_shots_made + total_shots_missed
    total_play_time = sum(s.duration_seconds for s in all_sessions)
    
    # Calculate average accuracy
    if total_sessions > 0:
        accuracies = [s.accuracy for s in all_sessions if s.total_shots > 0]
        average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    else:
        average_accuracy = 0.0
    
    # Current session stats
    current_session_stats = {
        'shots_made': current_stats['score'],
        'shots_missed': current_stats['misses'],
        'total_shots': current_stats['total_shots'],
        'accuracy': current_stats['accuracy']
    }
    
    return jsonify({
        'current_session': current_session_stats,
        'all_time': {
            'total_sessions': total_sessions,
            'total_shots_made': total_shots_made,
            'total_shots_missed': total_shots_missed,
            'total_shots': total_shots,
            'average_accuracy': round(average_accuracy, 1),
            'total_play_time_seconds': total_play_time,
            'total_play_time_formatted': format_play_time(total_play_time)
        },
        'camera_active': camera_active
    })

def format_play_time(seconds):
    """Format play time in a readable format"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"

@app.route('/api/camera/start', methods=['POST'])
@token_required
def start_camera():
    """Start the camera and tracking"""
    global camera_active, tracker, current_session, current_user
    user = request.current_user
    
    if not tracker:
        if not init_tracker():
            return jsonify({'error': 'Failed to initialize tracker'}), 500
    
    if not camera_active:
        if tracker.setup_camera():
            camera_active = True
            current_user = user
            
            # Create a new session
            current_session = Session(user_id=user.id)
            db.session.add(current_session)
            db.session.commit()
            
            # Reset tracker stats
            tracker.score = 0
            tracker.misses = 0
            tracker.ball_positions = []
            tracker.ball_in_hoop_zone = False
            
            update_stats()
            return jsonify({
                'status': 'success',
                'message': 'Camera started',
                'session_id': current_session.id
            })
        else:
            return jsonify({'error': 'Failed to start camera'}), 500
    else:
        return jsonify({'status': 'already_active', 'message': 'Camera is already active'})

@app.route('/api/camera/stop', methods=['POST'])
@token_required
def stop_camera():
    """Stop the camera and tracking"""
    global camera_active, tracker, current_session
    
    if tracker and camera_active:
        # Save session data before stopping
        if current_session:
            current_session.shots_made = tracker.score
            current_session.shots_missed = tracker.misses
            current_session.total_shots = tracker.score + tracker.misses
            current_session.calculate_accuracy()
            current_session.end_session()
            db.session.commit()
            current_session = None
        
        tracker.cleanup()
        camera_active = False
        update_stats()
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
    else:
        return jsonify({'status': 'already_stopped', 'message': 'Camera is not active'})

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    """Get camera status"""
    return jsonify({'active': camera_active})

@app.route('/api/reset', methods=['POST'])
@token_required
def reset_stats():
    """Reset tracking statistics"""
    global tracker, current_session
    if tracker:
        # Save current session before resetting
        if current_session and (tracker.score > 0 or tracker.misses > 0):
            current_session.shots_made = tracker.score
            current_session.shots_missed = tracker.misses
            current_session.total_shots = tracker.score + tracker.misses
            current_session.calculate_accuracy()
            current_session.end_session()
            db.session.commit()
            
            # Start a new session
            current_session = Session(user_id=request.current_user.id)
            db.session.add(current_session)
            db.session.commit()
        
        tracker.score = 0
        tracker.misses = 0
        tracker.ball_positions = []
        tracker.ball_in_hoop_zone = False
        update_stats()
        return jsonify({'status': 'success', 'message': 'Stats reset'})
    else:
        return jsonify({'error': 'Tracker not initialized'}), 500

@app.route('/api/manual/miss', methods=['POST'])
@token_required
def manual_miss():
    """Manually record a miss"""
    global tracker, current_session
    if tracker:
        tracker.record_miss(force=True)
        
        # Record shot in database
        if current_session:
            shot = Shot(session_id=current_session.id, made=False)
            db.session.add(shot)
            db.session.commit()
        
        update_stats()
        return jsonify({'status': 'success', 'message': 'Miss recorded'})
    else:
        return jsonify({'error': 'Tracker not initialized'}), 500

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get leaderboard with top players"""
    # Get all users with their total stats
    users = User.query.all()
    leaderboard_data = []
    
    for user in users:
        sessions = Session.query.filter_by(user_id=user.id).all()
        total_shots_made = sum(s.shots_made for s in sessions)
        total_shots = sum(s.total_shots for s in sessions)
        
        if total_shots > 0:
            accuracies = [s.accuracy for s in sessions if s.total_shots > 0]
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        else:
            avg_accuracy = 0.0
        
        leaderboard_data.append({
            'user_id': user.id,
            'username': user.username,
            'total_shots_made': total_shots_made,
            'total_shots': total_shots,
            'average_accuracy': round(avg_accuracy, 1)
        })
    
    # Sort by total shots made (descending)
    leaderboard_data.sort(key=lambda x: x['total_shots_made'], reverse=True)
    
    # Add ranks
    for i, entry in enumerate(leaderboard_data, 1):
        entry['rank'] = i
        entry['isTop3'] = i <= 3
    
    # Get top 10
    top_10 = leaderboard_data[:10]
    
    # Get current user's rank if authenticated
    user_rank = None
    top_score = leaderboard_data[0]['total_shots_made'] if leaderboard_data else 0
    total_players = len(leaderboard_data)
    
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            user_id = payload.get('user_id')
            for i, entry in enumerate(leaderboard_data, 1):
                if entry['user_id'] == user_id:
                    user_rank = i
                    break
        except:
            pass
    
    return jsonify({
        'players': top_10,
        'user_rank': user_rank,
        'top_score': top_score,
        'total_players': total_players
    })

def generate_frames():
    """Generator function for video streaming"""
    global tracker, camera_active
    
    while True:
        if tracker and camera_active and tracker.cap:
            ret, frame = tracker.cap.read()
            if ret:
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame with tracking
                annotated_frame = tracker.process_frame(frame)
                
                # Update stats (without database commit in video stream)
                update_stats(save_to_db=False)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                break
        else:
            # Send a placeholder frame when camera is not active
            placeholder = cv2.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, 'Camera Inactive', (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/api/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current tracking settings"""
    global tracker
    if tracker:
        return jsonify({
            'player_tracking_enabled': False,  # Not implemented in LiveCameraDetection yet
            'target_fps': tracker.target_fps,
            'device': tracker.device
        })
    else:
        return jsonify({'error': 'Tracker not initialized'}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update tracking settings"""
    global tracker
    if not tracker:
        return jsonify({'error': 'Tracker not initialized'}), 500
    
    data = request.get_json()
    
    # Player tracking not implemented in LiveCameraDetection yet
    # Settings endpoint kept for compatibility but no-op for now
    if 'player_tracking_enabled' in data:
        # Future: implement player tracking in LiveCameraDetection
        pass
    
    return jsonify({'status': 'success', 'message': 'Settings endpoint ready (player tracking coming soon)'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Courtvision API Server")
    print("="*60)
    print("Starting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("="*60 + "\n")
    
    # Initialize tracker
    init_tracker()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

