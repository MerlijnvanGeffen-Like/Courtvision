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
from basketball_tracking_system import BasketballTrackingSystem

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global tracking system instance
tracker = None
tracker_thread = None
camera_active = False
current_stats = {
    'score': 0,
    'misses': 0,
    'total_shots': 0,
    'accuracy': 0.0,
    'players_detected': 0,
    'camera_active': False
}

def init_tracker():
    """Initialize the basketball tracking system"""
    global tracker
    try:
        tracker = BasketballTrackingSystem('best.pt', enable_player_tracking=True)
        print("✓ Tracking system initialized")
        return True
    except Exception as e:
        print(f"✗ Error initializing tracker: {e}")
        return False

def update_stats():
    """Update current stats from tracker"""
    global current_stats, tracker
    if tracker:
        total_shots = tracker.score + tracker.misses
        accuracy = (tracker.score / total_shots * 100) if total_shots > 0 else 0.0
        
        current_stats = {
            'score': tracker.score,
            'misses': tracker.misses,
            'total_shots': total_shots,
            'accuracy': round(accuracy, 1),
            'players_detected': len(tracker.tracked_players) if tracker.enable_player_tracking else 0,
            'camera_active': camera_active
        }

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'API is running'})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get current tracking statistics"""
    update_stats()
    return jsonify(current_stats)

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start the camera and tracking"""
    global camera_active, tracker
    
    if not tracker:
        if not init_tracker():
            return jsonify({'error': 'Failed to initialize tracker'}), 500
    
    if not camera_active:
        if tracker.setup_camera():
            camera_active = True
            update_stats()
            return jsonify({'status': 'success', 'message': 'Camera started'})
        else:
            return jsonify({'error': 'Failed to start camera'}), 500
    else:
        return jsonify({'status': 'already_active', 'message': 'Camera is already active'})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop the camera and tracking"""
    global camera_active, tracker
    
    if tracker and camera_active:
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
def reset_stats():
    """Reset tracking statistics"""
    global tracker
    if tracker:
        tracker.score = 0
        tracker.misses = 0
        tracker.ball_positions = []
        tracker.ball_in_hoop_zone = False
        update_stats()
        return jsonify({'status': 'success', 'message': 'Stats reset'})
    else:
        return jsonify({'error': 'Tracker not initialized'}), 500

@app.route('/api/manual/miss', methods=['POST'])
def manual_miss():
    """Manually record a miss"""
    global tracker
    if tracker:
        tracker.record_miss(force=True)
        update_stats()
        return jsonify({'status': 'success', 'message': 'Miss recorded'})
    else:
        return jsonify({'error': 'Tracker not initialized'}), 500

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
                
                # Update stats
                update_stats()
                
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
            'player_tracking_enabled': tracker.enable_player_tracking,
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
    
    if 'player_tracking_enabled' in data:
        tracker.enable_player_tracking = data['player_tracking_enabled']
        if tracker.enable_player_tracking:
            tracker.target_classes = [0, 2, 3]
        else:
            tracker.target_classes = [0, 3]
    
    return jsonify({'status': 'success', 'message': 'Settings updated'})

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

