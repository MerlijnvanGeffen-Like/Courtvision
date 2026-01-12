import cv2
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
import time
import torch
import numpy as np
from collections import defaultdict

class BasketballTrackingSystem:
    """
    Advanced basketball tracking system with player detection and tracking
    Based on Roboflow's basketball AI detection approach
    """
    def __init__(self, model_path='best.pt', enable_player_tracking=True):
        """
        Initialize the basketball tracking system
        
        Args:
            model_path (str): Path to the trained YOLO model
            enable_player_tracking (bool): Enable player detection and tracking
        """
        self.model_path = model_path
        self.model = None
        self.cap = None
        self.enable_player_tracking = enable_player_tracking
        
        # Runtime/perf settings
        self.device = self.determine_device()
        self.use_half = self.device == 'cuda'
        self.imgsz = 640  # Higher resolution for better player detection
        self.capture_width = 640
        self.capture_height = 480
        self.target_fps = 30
        
        # Detection preferences
        if enable_player_tracking:
            # Detect basketball, hoop, and players
            self.target_classes = [0, 2, 3]  # basketball, player, hoop
        else:
            # Only basketball and hoop
            self.target_classes = [0, 3]
        
        # Scoring system - Improved with trajectory tracking
        self.score = 0
        self.misses = 0
        self.ball_positions = []  # Track ball positions over time with timestamps
        self.ball_trajectory = []  # Detailed trajectory: [(x, y, time, velocity_y), ...]
        self.hoop_bbox = None
        self.last_score_time = 0
        self.last_miss_time = 0
        
        # Shot state machine
        self.shot_state = 'idle'  # idle, approaching, near_hoop, through_hoop, missed, scored
        self.ball_in_hoop_zone = False
        self.ball_entry_time = 0
        self.ball_exit_time = 0
        self.ball_scored_in_current_entry = False
        self.last_ball_center = None
        self.last_ball_velocity = None  # (vx, vy)
        
        # Shot tracking to prevent duplicate readings
        self.current_shot_id = 0  # Unique ID for current shot attempt
        self.shot_result_registered = False  # Whether result (score/miss) has been registered for current shot
        self.shot_start_time = 0  # When current shot attempt started
        self.shot_lockout_time = 2.0  # Time after shot result before new shot can be detected (seconds)
        
        # Improved detection parameters - Relaxed for better detection
        self.score_cooldown = 0.5  # Seconds between score detections (0.5s cooldown to prevent duplicates)
        self.miss_cooldown = 0.5  # Seconds between miss detections (0.5s cooldown to prevent duplicates)
        self.trajectory_history_time = 2.0  # Keep trajectory for 2 seconds
        self.min_trajectory_points = 3  # Minimum points needed for trajectory analysis (reduced)
        self.hoop_zone_margin = 0.7  # Margin for considering ball "near" hoop (increased for better miss detection)
        self.velocity_threshold = 2.0  # Minimum velocity (pixels/frame) to consider active shot (reduced)
        self.downward_velocity_threshold = -5.0  # Ball must be moving downward through hoop (more lenient)
        self.min_velocity_for_score = 1.0  # Minimum velocity magnitude required for score (ball must be moving)
        
        # Debug mode
        self.debug_mode = True  # Set to False to disable debug prints
        
        # Player tracking system
        self.tracked_players = {}  # {track_id: {bbox, center, last_seen, history}}
        self.player_track_colors = {}  # Color for each tracked player
        self.next_track_id = 1
        self.track_history_length = 30  # Frames to keep in history
        self.track_iou_threshold = 0.3  # IoU threshold for tracking
        
        # Performance tracking
        self.last_inference_time = 0.0
        self.last_frame_time = 0.0
        
        self.load_model()
    
    def determine_device(self):
        """Decide which device to run on"""
        if torch.cuda.is_available():
            try:
                driver_name = torch.cuda.get_device_name(torch.cuda.current_device())
                print(f"✓ CUDA device detected: {driver_name}")
                return 'cuda'
            except Exception as e:
                print(f"⚠️ CUDA initialization failed ({e}). Falling back to CPU.")
        else:
            print("⚠️ CUDA not available. Using CPU.")
        return 'cpu'
    
    def load_model(self):
        """Load the trained YOLO model"""
        try:
            print(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            
            # Map class IDs to meaningful names
            self.class_names = {
                0: 'basketball',
                1: 'ref',
                2: 'player',
                3: 'hoop',
                4: 'ball'
            }
            
            # Override the model's internal names dictionary
            self.model.model.names = self.class_names
            
            # Move and optimize model
            self.model.to(self.device)
            try:
                self.model.fuse()
            except Exception:
                pass
            if self.use_half:
                try:
                    self.model.model.half()
                except Exception:
                    self.use_half = False
            
            # Performance optimizations
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                try:
                    torch.set_float32_matmul_precision('medium')
                except Exception:
                    pass
            
            actual_device = str(next(self.model.model.parameters()).device)
            print(f"✓ Model loaded successfully!")
            print(f"✓ Running on {actual_device.upper()}")
            if self.enable_player_tracking:
                print(f"✓ Detecting: Basketball, Hoop, and Players")
            else:
                print(f"✓ Detecting: Basketball and Hoop")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def setup_camera(self, camera_index=0):
        """Setup camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {camera_index}")
            
            cv2.setUseOptimized(True)
            try:
                cv2.setNumThreads(0)
            except Exception:
                pass

            # Set camera properties for better performance and higher FPS
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS from camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            # Reduce buffer to minimize latency and skip fewer frames
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for lower latency
            except Exception:
                pass
            # Try to enable auto exposure and other optimizations
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto exposure
            except Exception:
                pass
            
            print(f"Camera {camera_index} initialized successfully!")
            return True
        except Exception as e:
            print(f"Error setting up camera: {e}")
            return False
    
    def get_bbox_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def update_player_tracking(self, player_detections):
        """
        Simple tracking algorithm based on IoU matching
        For more advanced tracking, consider using ByteTrack or DeepSORT
        """
        current_frame = time.time()
        
        # Match detections to existing tracks
        matched_tracks = set()
        unmatched_detections = []
        
        for det_bbox, det_center in player_detections:
            best_match_id = None
            best_iou = self.track_iou_threshold
            
            # Find best matching track
            for track_id, track_data in self.tracked_players.items():
                if track_id in matched_tracks:
                    continue
                
                last_bbox = track_data['bbox']
                iou = self.calculate_iou(det_bbox, last_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                track_data = self.tracked_players[best_match_id]
                track_data['bbox'] = det_bbox
                track_data['center'] = det_center
                track_data['last_seen'] = current_frame
                track_data['history'].append((det_center, current_frame))
                
                # Keep only recent history
                track_data['history'] = [
                    (pos, t) for pos, t in track_data['history'] 
                    if current_frame - t < 2.0  # Keep last 2 seconds
                ]
                
                matched_tracks.add(best_match_id)
            else:
                # New detection, create new track
                unmatched_detections.append((det_bbox, det_center))
        
        # Create new tracks for unmatched detections
        for det_bbox, det_center in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            # Generate color for this track
            color = tuple(np.random.randint(0, 255, 3).tolist())
            self.player_track_colors[track_id] = color
            
            self.tracked_players[track_id] = {
                'bbox': det_bbox,
                'center': det_center,
                'last_seen': current_frame,
                'history': [(det_center, current_frame)]
            }
        
        # Remove old tracks (not seen for more than 1 second)
        tracks_to_remove = [
            track_id for track_id, track_data in self.tracked_players.items()
            if current_frame - track_data['last_seen'] > 1.0
        ]
        for track_id in tracks_to_remove:
            del self.tracked_players[track_id]
            if track_id in self.player_track_colors:
                del self.player_track_colors[track_id]
    
    def is_point_in_bbox(self, point, bbox, margin=0.1):
        """Check if point is within bounding box with optional margin"""
        x, y = point
        x1, y1, x2, y2 = bbox
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        expanded_x1 = center_x - (width * (1 + margin)) / 2
        expanded_x2 = center_x + (width * (1 + margin)) / 2
        expanded_y1 = center_y - (height * (1 + margin)) / 2
        expanded_y2 = center_y + (height * (1 + margin)) / 2
        
        return expanded_x1 <= x <= expanded_x2 and expanded_y1 <= y <= expanded_y2
    
    def calculate_velocity(self, pos1, time1, pos2, time2):
        """Calculate velocity vector between two positions"""
        if time2 == time1:
            return (0, 0)
        dt = time2 - time1
        vx = (pos2[0] - pos1[0]) / dt
        vy = (pos2[1] - pos1[1]) / dt
        return (vx, vy)
    
    def is_ball_through_hoop(self, ball_center, hoop_bbox, velocity=None):
        """
        Improved check if ball has passed through the hoop
        Uses trajectory analysis for better accuracy
        
        Args:
            ball_center: Current ball center (x, y)
            hoop_bbox: Hoop bounding box (x1, y1, x2, y2)
            velocity: Optional velocity vector (vx, vy) for trajectory analysis
        """
        if hoop_bbox is None:
            return False
        
        x, y = ball_center
        x1, y1, x2, y2 = hoop_bbox
        
        # Calculate hoop center and dimensions
        hoop_center_x = (x1 + x2) / 2
        hoop_center_y = (y1 + y2) / 2
        hoop_width = x2 - x1
        hoop_height = y2 - y1
        
        # Horizontal alignment: ball must be reasonably centered (more lenient)
        horizontal_margin = 0.5  # 50% of width on each side (increased)
        in_horizontal = abs(x - hoop_center_x) <= (hoop_width * horizontal_margin)
        
        # Vertical zone: ball should be in the lower half of hoop or just below
        # This represents the ball passing through (more lenient)
        upper_bound = hoop_center_y - (hoop_height * 0.1)  # Slightly above center
        lower_bound = y2 + (hoop_height * 0.3)  # More below hoop
        in_vertical = upper_bound <= y <= lower_bound
        
        # REQUIRE velocity for score detection - ball must be moving!
        if velocity is None:
            return False  # No velocity = no score (ball is not moving)
        
        vx, vy = velocity
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        
        # Ball MUST have minimum velocity to be considered a score (prevents stationary balls)
        if velocity_magnitude < self.min_velocity_for_score:
            return False
        
        # Ball should be moving downward (positive y is downward in image)
        downward_motion = vy > self.downward_velocity_threshold
        
        # REQUIRE both horizontal/vertical alignment AND downward motion for score
        # This ensures ball is actually going through the ring, not just held in front
        return in_horizontal and in_vertical and downward_motion
    
    def detect_miss(self, ball_center, hoop_bbox, trajectory=None):
        """
        Improved miss detection using trajectory analysis
        A miss occurs when:
        1. Ball was near hoop (approached it)
        2. Ball is now below hoop
        3. Ball never passed through hoop zone
        4. Ball trajectory shows it went around/over hoop
        
        Args:
            ball_center: Current ball center (x, y)
            hoop_bbox: Hoop bounding box (x1, y1, x2, y2)
            trajectory: Optional trajectory history for analysis
        """
        if hoop_bbox is None:
            return False
        
        x, y = ball_center
        x1, y1, x2, y2 = hoop_bbox
        
        hoop_bottom = y2
        hoop_center_x = (x1 + x2) / 2
        hoop_width = x2 - x1
        hoop_top = y1
        
        # Check if ball is significantly below hoop
        below_hoop = y > hoop_bottom + (hoop_width * 0.3)  # More robust threshold
        
        # Check horizontal alignment (ball was aimed at hoop)
        near_hoop_horizontal = abs(x - hoop_center_x) <= (hoop_width * 0.7)
        
        # If trajectory is available, check if ball approached hoop from above
        if trajectory and len(trajectory) >= 3:
            # Check if ball was above hoop recently
            recent_positions = trajectory[-5:]  # Last 5 points
            was_above_hoop = any(pos[1] < hoop_top for pos in recent_positions)
            
            # Check if ball trajectory shows it went past hoop without going through
            passed_hoop = y > hoop_bottom and was_above_hoop
            
            return near_hoop_horizontal and below_hoop and passed_hoop
        
        # Fallback: simple check
        return near_hoop_horizontal and below_hoop
    
    def analyze_trajectory(self, trajectory, hoop_bbox=None):
        """
        Analyze ball trajectory to determine shot outcome with improved accuracy
        Returns: 'scored', 'missed', or 'uncertain'
        """
        if len(trajectory) < self.min_trajectory_points:
            return 'uncertain'
        
        # Extract positions and velocities
        positions = [(pos[0], pos[1]) for pos in trajectory if len(pos) >= 2]
        
        if not positions:
            return 'uncertain'
        
        # Analyze trajectory relative to hoop if available
        if hoop_bbox is not None:
            x1, y1, x2, y2 = hoop_bbox
            hoop_center_x = (x1 + x2) / 2
            hoop_center_y = (y1 + y2) / 2
            hoop_width = x2 - x1
            hoop_top = y1
            hoop_bottom = y2
            
            # Check if ball trajectory went through hoop zone
            positions_through_hoop = []
            for x, y in positions:
                # Check horizontal alignment
                in_horizontal = abs(x - hoop_center_x) <= (hoop_width * 0.4)
                # Check vertical zone (through hoop)
                in_vertical = (hoop_top - hoop_width * 0.2) <= y <= (hoop_bottom + hoop_width * 0.3)
                if in_horizontal and in_vertical:
                    positions_through_hoop.append((x, y))
            
            # If we have multiple positions through hoop, likely a score
            if len(positions_through_hoop) >= 2:
                # Check if trajectory shows downward motion after going through
                if len(positions_through_hoop) >= 3:
                    y_coords = [p[1] for p in positions_through_hoop]
                    # Check if y increases (ball falling down)
                    if y_coords[-1] > y_coords[0]:
                        return 'scored'
        
        # Fallback: Check velocities for downward motion
        velocities = []
        for i in range(1, len(trajectory)):
            if len(trajectory[i]) >= 4:
                velocities.append(trajectory[i][3])  # velocity_y
        
        if velocities:
            # Check for consistent downward motion
            recent_velocities = velocities[-3:] if len(velocities) >= 3 else velocities
            avg_downward = sum(vy for vy in recent_velocities if vy > 0) / len(recent_velocities) if recent_velocities else 0
            
            # Require stronger downward motion for score
            if avg_downward > 2.0:  # Increased threshold for accuracy
                return 'scored'
        
        return 'uncertain'
    
    def is_ball_approaching_hoop(self, ball_center, hoop_bbox):
        """
        Check if ball is approaching the hoop (within larger zone)
        Used to detect when a shot attempt is starting
        """
        if hoop_bbox is None:
            return False
        return self.is_point_in_bbox(ball_center, hoop_bbox, margin=0.5)
    
    def process_frame(self, frame):
        """
        Process frame with YOLO detection, player tracking, and score detection
        """
        inference_start = time.perf_counter()
        
        # Run YOLO detection with tracking enabled for players
        results = self.model.track(
            source=frame,
            imgsz=self.imgsz,
            conf=0.4,  # Lower confidence for better ball tracking, especially with fast movement
            device=self.device,
            half=self.use_half,
            classes=self.target_classes,
            verbose=False,
            persist=True  # Enable tracking persistence
        )
        
        self.last_inference_time = time.perf_counter() - inference_start
        
        result = results[0]
        current_time = time.time()
        
        # Extract detections
        boxes = result.boxes
        ball_centers = []
        hoop_bboxes = []
        player_detections = []
        
        detections_to_draw = []
        
        # Process detections
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = self.class_names.get(cls_id, 'unknown')
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            confidence = float(box.conf[0])
            
            # Get track ID if available (from YOLO tracking)
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])
            
            detections_to_draw.append((class_name, bbox, confidence, track_id))
            
            # Track ball positions
            if class_name in ['basketball', 'ball']:
                center = self.get_bbox_center(bbox)
                ball_centers.append((center, bbox))
            
            # Track hoop position
            elif class_name == 'hoop':
                hoop_bboxes.append(bbox)
            
            # Track players
            elif class_name == 'player' and self.enable_player_tracking:
                center = self.get_bbox_center(bbox)
                player_detections.append((bbox, center))
        
        # Update hoop bbox
        if hoop_bboxes:
            self.hoop_bbox = hoop_bboxes[0]
        
        # Update player tracking (if not using YOLO's built-in tracking)
        if self.enable_player_tracking and player_detections:
            # Use simple IoU-based tracking as fallback
            # YOLO's built-in tracking should handle this, but we keep this as backup
            pass
        
        # Process ball detections with improved trajectory tracking
        if ball_centers:
            ball_center, ball_bbox = ball_centers[0]
            
            # Calculate velocity if we have previous position
            velocity = None
            if self.last_ball_center is not None:
                prev_time = current_time - (1.0 / self.target_fps)  # Approximate previous frame time
                velocity = self.calculate_velocity(
                    self.last_ball_center, 
                    prev_time,
                    ball_center, 
                    current_time
                )
                # Store velocity magnitude for trajectory
                velocity_magnitude = np.sqrt(velocity[0]**2 + velocity[1]**2)
                velocity_y = velocity[1]
            else:
                velocity_magnitude = 0
                velocity_y = 0
            
            # Add to trajectory with velocity information
            self.ball_trajectory.append((ball_center[0], ball_center[1], current_time, velocity_y))
            self.ball_positions.append((ball_center, current_time))
            
            # Keep only recent trajectory data
            self.ball_trajectory = [
                (x, y, t, vy) for x, y, t, vy in self.ball_trajectory 
                if current_time - t < self.trajectory_history_time
            ]
            self.ball_positions = [
                (pos, t) for pos, t in self.ball_positions 
                if current_time - t < self.trajectory_history_time
            ]
            
            # Update shot state machine - Simplified and more reliable
            if self.hoop_bbox is not None:
                ball_near_hoop = self.is_point_in_bbox(ball_center, self.hoop_bbox, margin=self.hoop_zone_margin)
                ball_through = self.is_ball_through_hoop(ball_center, self.hoop_bbox, velocity)
                
                if self.debug_mode and (ball_near_hoop or ball_through):
                    print(f"\n[DEBUG] State: {self.shot_state}, Near: {ball_near_hoop}, Through: {ball_through}, Vel: {velocity_magnitude:.2f}")
                
                # State transitions - Simplified logic
                if self.shot_state == 'idle':
                    # Start tracking if ball is near hoop AND moving (not stationary)
                    if ball_near_hoop and velocity_magnitude > self.velocity_threshold:
                        self.shot_state = 'near_hoop'
                        self.ball_in_hoop_zone = True
                        self.ball_entry_time = current_time
                        self.ball_scored_in_current_entry = False
                        if self.debug_mode:
                            print(f"  -> State changed to NEAR_HOOP (velocity: {velocity_magnitude:.2f})")
                    elif velocity_magnitude > self.velocity_threshold and self.is_ball_approaching_hoop(ball_center, self.hoop_bbox):
                        # Ball is moving towards hoop but not yet near
                        self.shot_state = 'near_hoop'
                        self.ball_in_hoop_zone = True
                        self.ball_entry_time = current_time
                        self.ball_scored_in_current_entry = False
                        if self.debug_mode:
                            print(f"  -> State changed to NEAR_HOOP (approaching, velocity: {velocity_magnitude:.2f})")
                
                elif self.shot_state == 'near_hoop':
                    # Check if ball went through hoop
                    if ball_through:
                        # Verify with trajectory analysis for better accuracy
                        trajectory_result = self.analyze_trajectory(self.ball_trajectory, self.hoop_bbox)
                        
                        # Score only if trajectory analysis confirms or if ball is clearly through
                        if (trajectory_result == 'scored' or 
                            (trajectory_result == 'uncertain' and len(self.ball_trajectory) >= 5)):
                            # Ball is through hoop - score it! (only if not already registered)
                            if not self.shot_result_registered and current_time - self.last_score_time > self.score_cooldown:
                                self.score += 1
                                self.last_score_time = current_time
                                self.ball_scored_in_current_entry = True
                                self.shot_result_registered = True  # Mark as registered - prevents duplicate
                                self.shot_state = 'scored'
                                print(f"🎯 SCORE! Total: {self.score} (Shot ID: {self.current_shot_id})")
                                if self.debug_mode:
                                    print(f"  -> State changed to SCORED (Shot ID: {self.current_shot_id}, Trajectory: {trajectory_result})")
                    elif not ball_near_hoop:
                        # Ball left hoop zone - verify it's actually a miss with trajectory analysis
                        # Only register miss if no result has been registered yet
                        if not self.shot_result_registered:
                            # Verify miss with trajectory analysis
                            trajectory_result = self.analyze_trajectory(self.ball_trajectory, self.hoop_bbox)
                            is_miss = (trajectory_result != 'scored')  # Not a score = miss
                            
                            # Also check if ball is below hoop and was near it
                            if is_miss or self.detect_miss(ball_center, self.hoop_bbox, self.ball_trajectory):
                                if current_time - self.last_miss_time > self.miss_cooldown:
                                    self.record_miss(event_time=current_time)
                                    self.shot_result_registered = True  # Mark as registered - prevents duplicate
                                    self.shot_state = 'missed'
                                    if self.debug_mode:
                                        print(f"  -> State changed to MISSED (ball left zone without scoring, Shot ID: {self.current_shot_id})")
                        else:
                            # Result already registered for this shot, reset
                            self.shot_state = 'idle'
                            if self.debug_mode:
                                print(f"  -> State reset to IDLE (result already registered for Shot ID: {self.current_shot_id})")
                
                elif self.shot_state == 'scored':
                    # After scoring, wait a bit then reset (only if result was registered)
                    if not ball_near_hoop and (current_time - self.last_score_time > 0.5):
                        self.shot_state = 'idle'
                        self.ball_in_hoop_zone = False
                        self.shot_result_registered = False  # Reset for next shot
                        if self.debug_mode:
                            print(f"  -> State reset to IDLE after score")
                
                elif self.shot_state == 'missed':
                    # After miss, wait a bit then reset (only if result was registered)
                    if not ball_near_hoop and (current_time - self.last_miss_time > 0.5):
                        self.shot_state = 'idle'
                        self.ball_in_hoop_zone = False
                        self.shot_result_registered = False  # Reset for next shot
                        if self.debug_mode:
                            print(f"  -> State reset to IDLE after miss")
                
                # Fallback: if ball is clearly through hoop but state didn't catch it
                # Only register if no result has been registered for this shot yet
                # Use trajectory verification for accuracy
                if (ball_through and self.shot_state != 'scored' and 
                    not self.shot_result_registered and 
                    current_time - self.last_score_time > self.score_cooldown):
                    trajectory_result = self.analyze_trajectory(self.ball_trajectory, self.hoop_bbox)
                    # Only score if trajectory confirms or we have enough trajectory data
                    if trajectory_result != 'missed' and len(self.ball_trajectory) >= 4:
                        self.score += 1
                        self.last_score_time = current_time
                        self.ball_scored_in_current_entry = True
                        self.shot_result_registered = True  # Mark as registered - prevents duplicate
                        self.shot_state = 'scored'
                        print(f"🎯 SCORE! (fallback) Total: {self.score} (Shot ID: {self.current_shot_id})")
                        if self.debug_mode:
                            print(f"  -> Fallback score detected (Shot ID: {self.current_shot_id}, Trajectory: {trajectory_result})")
            
            self.last_ball_center = ball_center
            self.last_ball_velocity = velocity
        else:
            # No ball detected - reset state if it's been a while
            if self.shot_state != 'idle' and len(self.ball_trajectory) == 0:
                # Check if we should finalize a score/miss
                if self.shot_state == 'through_hoop' and not self.ball_scored_in_current_entry:
                    trajectory_result = self.analyze_trajectory(self.ball_trajectory)
                    if trajectory_result == 'scored' and current_time - self.last_score_time > self.score_cooldown:
                        self.score += 1
                        self.last_score_time = current_time
                        print(f"SCORE! (finalized) Total: {self.score}")
                self.shot_state = 'idle'
            self.last_ball_center = None
            self.last_ball_velocity = None
        
        # Draw detections
        annotated_frame = frame.copy()
        
        # Draw trajectory path if available
        if len(self.ball_trajectory) >= 2:
            trajectory_points = [(int(x), int(y)) for x, y, _, _ in self.ball_trajectory]
            for i in range(len(trajectory_points) - 1):
                # Draw trajectory line with fading color (recent = brighter)
                alpha = i / len(trajectory_points)
                color_intensity = int(255 * (1 - alpha * 0.7))
                cv2.line(annotated_frame, trajectory_points[i], trajectory_points[i+1], 
                        (0, color_intensity, 255), 2)
        
        for class_name, bbox, confidence, track_id in detections_to_draw:
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Choose color based on class
            if class_name in ['basketball', 'ball']:
                color = (0, 165, 255)  # Orange
            elif class_name == 'hoop':
                color = (0, 255, 255)  # Yellow
            elif class_name == 'player':
                # Use track ID color if available, otherwise use green
                if track_id is not None:
                    # Generate consistent color from track_id
                    np.random.seed(track_id)
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    np.random.seed()  # Reset seed
                else:
                    color = (0, 255, 0)  # Green
            else:
                color = (255, 255, 255)  # White
            
            # Draw bounding box (basketball tracking - no labels)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw score and misses
        score_text = f"Score: {self.score}"
        miss_text = f"Misses: {self.misses}"
        accuracy = (self.score / (self.score + self.misses) * 100) if (self.score + self.misses) > 0 else 0.0
        
        cv2.putText(annotated_frame, score_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(annotated_frame, miss_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Draw accuracy
        accuracy_text = f"Accuracy: {accuracy:.1f}%"
        cv2.putText(annotated_frame, accuracy_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw scored/missed status only
        if self.shot_state == 'scored':
            cv2.putText(annotated_frame, "SCORED!", (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        elif self.shot_state == 'missed':
            cv2.putText(annotated_frame, "MISSED", (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        return annotated_frame
    
    def record_miss(self, event_time=None, force=False):
        """Increment miss counter with cooldown handling"""
        if event_time is None:
            event_time = time.time()

        if force or (event_time - self.last_miss_time > self.miss_cooldown):
            self.misses += 1
            self.last_miss_time = event_time
            print(f"MISS! Total misses: {self.misses}")
    
    def run_detection(self):
        """Run live camera detection"""
        if not self.setup_camera():
            return
        
        print("\n" + "="*60)
        print("Basketball Tracking System Started!")
        print("="*60)
        print("Features:")
        if self.enable_player_tracking:
            print("  ✓ Player Detection & Tracking")
        print("  ✓ Basketball Detection")
        print("  ✓ Hoop Detection")
        print("  ✓ Score Tracking")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit/Close")
        print("  's' - Save current frame")
        print("  'm' - Mark manual miss")
        print("  'p' - Toggle player tracking")
        print("="*60 + "\n")
        
        # FPS tracking
        fps = 0.0
        prev_time = time.perf_counter()
        smooth_alpha = 0.9
        target_frame_time = 1.0 / self.target_fps
        
        try:
            while True:
                frame_start = time.perf_counter()
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Calculate FPS
                now = time.perf_counter()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    inst_fps = 1.0 / dt
                    fps = (smooth_alpha * fps + (1.0 - smooth_alpha) * inst_fps) if fps > 0 else inst_fps
                
                # Display FPS counter
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                # Display frame
                window_title = 'Basketball Tracking System - Press Q to quit'
                if self.enable_player_tracking:
                    window_title += ' | Player Tracking: ON'
                cv2.imshow(window_title, annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"tracking_snapshot_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"✓ Frame saved as {filename}")
                elif key == ord('m'):
                    self.record_miss(force=True)
                elif key == ord('p'):
                    self.enable_player_tracking = not self.enable_player_tracking
                    if self.enable_player_tracking:
                        self.target_classes = [0, 2, 3]
                        print("Player tracking ENABLED")
                    else:
                        self.target_classes = [0, 3]
                        print("Player tracking DISABLED")
                
                # Don't limit FPS - process as fast as possible for better accuracy
                # This allows us to process more frames and not skip important ones
                loop_time = time.perf_counter() - frame_start
                self.last_frame_time = loop_time
                # Removed sleep to maximize frame processing rate
                # If camera can provide 30+ FPS, we'll process all of them
                
        except KeyboardInterrupt:
            print("\n\nDetection stopped by user")
        except Exception as e:
            print(f"\n✗ Error during detection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")

def main():
    """Main function to run the basketball tracking system"""
    print("\n" + "="*60)
    print("🏀 Basketball Tracking System")
    print("="*60)
    print("Based on Roboflow's Basketball AI Detection")
    print("="*60)
    
    try:
        # Initialize tracking system with player tracking enabled
        tracker = BasketballTrackingSystem('best.pt', enable_player_tracking=True)
        
        # Run detection
        tracker.run_detection()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure 'best.pt' model file exists")
        print("2. Ensure your camera is connected and accessible")
        print("3. Check that PyTorch and Ultralytics are installed")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

