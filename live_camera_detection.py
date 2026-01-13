import cv2
from ultralytics import YOLO
import time
import torch
import numpy as np

class LiveCameraDetection:
    def __init__(self, model_path='best.pt'):
        """
        Initialize the live camera detection system
        
        Args:
            model_path (str): Path to the trained YOLO model
        """
        self.model_path = model_path
        self.model = None
        self.cap = None
        # Runtime/perf settings
        self.device = self.determine_device()
        self.use_half = self.device == 'cuda'
        self.imgsz = 384  # Increased for better accuracy (was 288, now 384 for higher quality)
        self.capture_width = 640  # Increased from 480 for better quality and wider FOV
        self.capture_height = 360  # Increased from 270 for better quality and wider FOV (16:9 aspect ratio)
        self.target_fps = 30  # Target FPS for processing (already at 30)
        self.inference_mode = torch.no_grad  # Will be set to inference_mode if available
        self.roi_enabled = False  # Disabled: track entire frame instead of just ROI around hoop
        self.roi_margin = 140
        self.roi_min_size = 200
        self.roi_reset_frames = 40
        self.frames_since_hoop = self.roi_reset_frames
        self.current_roi = None  # (x1, y1, x2, y2)
        
        # Detection preferences (basketball + hoop only)
        self.target_classes = [0, 3]

        # Scoring system - Improved with trajectory tracking
        self.score = 0
        self.misses = 0
        self.ball_positions = []  # Track ball positions over time with timestamps
        self.ball_trajectory = []  # Detailed trajectory: [(x, y, time, velocity_y), ...]
        self.hoop_bbox = None  # Current hoop bounding box
        self.last_score_time = 0  # Prevent duplicate scoring
        self.last_miss_time = 0  # Prevent duplicate miss detection
        self.last_hoop_detection_time = 0  # Time of last hoop detection
        self.hoop_detection_interval = 5.0  # Detect hoop every 5 seconds (ring doesn't move)
        
        # Shot state machine - Simplified system
        self.shot_state = 'idle'  # idle, shooting, scored, missed
        self.ball_in_hoop_zone = False  # Track if ball is near hoop
        self.ball_entry_time = 0
        self.ball_exit_time = 0
        self.ball_scored_in_current_entry = False
        self.last_ball_center = None
        self.last_ball_velocity = None  # (vx, vy)
        self.frames_without_ball = 0  # Track consecutive frames without ball detection
        self.max_prediction_frames = 15  # Maximum frames to predict trajectory when ball not detected (increased for better tracking near ring)
        self.score_verification_frames = []  # Track frames where ball appears to go through hoop
        self.trajectory_smooth_window = 3  # Number of points to use for trajectory smoothing
        self.min_velocity_for_score = 1.0  # Minimum velocity magnitude required for score (ball must be moving)
        self.ball_entered_hoop_zone = False  # Track if ball entered hoop zone from above
        
        # Shot tracking to prevent duplicate readings
        self.current_shot_id = 0  # Unique ID for current shot attempt
        self.shot_result_registered = False  # Whether result (score/miss) has been registered for current shot
        self.shot_start_time = 0  # When current shot attempt started
        self.shot_lockout_time = 2.0  # Time after shot result before new shot can be detected (seconds)
        self.trajectory_length_at_result = None  # Track trajectory length when result was registered (prevents duplicate results for same trajectory)
        
        # Improved detection parameters - Relaxed for better detection
        self.score_cooldown = 1.0  # Seconds between score detections (1.0s cooldown to prevent duplicates)
        self.miss_cooldown = 0.5  # Seconds between miss detections (0.5s cooldown to prevent duplicates)
        self.trajectory_history_time = 2.0  # Keep trajectory for 2 seconds
        self.min_trajectory_points = 3  # Minimum points needed for trajectory analysis (reduced)
        self.hoop_zone_margin = 0.7  # Margin for considering ball "near" hoop (increased for better miss detection)
        self.velocity_threshold = 2.0  # Minimum velocity (pixels/frame) to consider active shot (reduced)
        self.downward_velocity_threshold = -5.0  # Ball must be moving downward through hoop (more lenient)
        
        # Two-box system: Shot box (larger) and Score box (smaller, inside shot box)
        self.shot_box_horizontal_margin = 0.8  # 80% margin for shot box (larger area for shot attempts)
        self.shot_box_vertical_top_margin = 0.7  # 70% above hoop center (extends well above ring to catch rim shots)
        self.shot_box_vertical_bottom_margin = 0.6  # 60% below hoop bottom
        
        self.score_box_horizontal_margin = 0.5  # 50% margin for score box (smaller, precise area for scores)
        self.score_box_vertical_top_margin = 0.0  # Start at hoop center (moved down, less height)
        self.score_box_vertical_bottom_margin = 0.5  # 50% below hoop bottom (moved down to sit under net)
        
        # Debug mode
        self.debug_mode = False  # Set to False for better FPS (disable debug prints)
        
        # Performance tracking
        self.last_inference_time = 0.0
        self.last_frame_time = 0.0
        
        self.load_model()
        
    def determine_device(self):
        """Decide which device to run on and explain fallback reasons"""
        if torch.cuda.is_available():
            try:
                driver_name = torch.cuda.get_device_name(torch.cuda.current_device())
                print(f"✓ CUDA device detected: {driver_name}")
                return 'cuda'
            except Exception as e:
                print(f"⚠️ CUDA initialization failed ({e}). Falling back to CPU.")
        else:
            print("⚠️ CUDA not available (PyTorch without GPU support or driver missing). Using CPU.")
        return 'cpu'
    
    def load_model(self):
        """Load the trained YOLO model with advanced optimizations"""
        try:
            print(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            
            # Map class IDs to meaningful names for basketball detection
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
            
            # Model fusion (combines conv+bn layers for faster inference)
            try:
                self.model.fuse()
                print("✓ Model fused (conv+bn layers combined)")
            except Exception:
                pass
            
            # Half precision (FP16) for GPU - 2x faster inference
            if self.use_half:
                try:
                    self.model.model.half()
                    print("✓ Half precision (FP16) enabled")
                except Exception:
                    self.use_half = False
                    print("⚠ Half precision not available, using FP32")
            
            # Set model to evaluation mode (disables dropout, batch norm updates)
            self.model.model.eval()
            
            # Extra backend perf tweaks
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
                print("✓ cuDNN benchmark enabled")
            
            # PyTorch 2.0+ optimizations
            if hasattr(torch, "set_float32_matmul_precision"):
                try:
                    torch.set_float32_matmul_precision('medium')  # or 'high' for accuracy
                    print("✓ Float32 matmul precision optimized")
                except Exception:
                    pass
            
            # Enable inference mode (faster than no_grad, disables autograd completely)
            if hasattr(torch, "inference_mode"):
                self.inference_mode = torch.inference_mode
            else:
                self.inference_mode = torch.no_grad
            
            # Threading optimizations for CPU
            if self.device == 'cpu':
                torch.set_num_threads(4)  # Optimize for 4 cores (adjust based on your CPU)
                torch.set_num_interop_threads(2)
                print("✓ CPU threading optimized")
            
            # Model warmup - run dummy inference to optimize model
            print("Warming up model...")
            dummy_frame = np.zeros((self.capture_height, self.capture_width, 3), dtype=np.uint8)
            try:
                with self.inference_mode():
                    _ = self.model.predict(
                        source=dummy_frame,
                        imgsz=self.imgsz,
                        conf=0.58,  # Increased by 15% (was 0.50, now 0.58) for better ball and hoop detection
                        device=self.device,
                        half=self.use_half,
                        verbose=False
                    )
                print("✓ Model warmup completed")
            except Exception as e:
                print(f"⚠ Warmup failed (non-critical): {e}")
            
            actual_device = str(next(self.model.model.parameters()).device)
            print(f"✓ Model loaded successfully!")
            print(f"✓ Running on {actual_device.upper()}")
            print(f"✓ Detecting: Basketball and Hoop")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def setup_camera(self, camera_index=0):
        """
        Setup camera capture
        
        Args:
            camera_index (int): Camera index (0 for default camera)
        """
        try:
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {camera_index}")
            
            # OpenCV runtime optimizations
            cv2.setUseOptimized(True)
            try:
                cv2.setNumThreads(4)  # Use 4 threads for OpenCV (adjust based on CPU)
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
    
    def enhance_frame(self, frame):
        """Hook for future brightness/contrast tweaks (currently passthrough)"""
        return frame
    
    def get_bbox_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def is_ball_approaching_hoop(self, ball_center, hoop_bbox):
        """
        Check if ball is approaching the hoop (within larger zone)
        Used to detect when a shot attempt is starting
        """
        if hoop_bbox is None:
            return False
        return self.is_point_in_bbox(ball_center, hoop_bbox, margin=0.5)
    
    def is_point_in_bbox(self, point, bbox, margin=0.1):
        """
        Check if point is within bounding box with optional margin
        
        Args:
            point: (x, y) tuple
            bbox: (x1, y1, x2, y2) bounding box
            margin: Percentage margin to expand bbox (0.1 = 10% larger)
        """
        x, y = point
        x1, y1, x2, y2 = bbox
        
        # Calculate center and dimensions
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Expand bbox by margin
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
    
    def is_ball_in_shot_box(self, ball_center, hoop_bbox, velocity=None):
        """
        SIMPLE: Check if ball is in the SHOT BOX (green box)
        Only checks position - no velocity or other checks
        
        Args:
            ball_center: Current ball center (x, y)
            hoop_bbox: Hoop bounding box (x1, y1, x2, y2)
            velocity: Ignored - kept for compatibility
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
        
        # Shot box: larger horizontal margin
        in_horizontal = abs(x - hoop_center_x) <= (hoop_width * self.shot_box_horizontal_margin)
        
        # Shot box: larger vertical zone
        upper_bound = hoop_center_y - (hoop_height * self.shot_box_vertical_top_margin)
        lower_bound = y2 + (hoop_height * self.shot_box_vertical_bottom_margin)
        in_vertical = upper_bound <= y <= lower_bound
        
        # SIMPLE: Just check if ball is in the box - nothing else
        return in_horizontal and in_vertical
    
    def is_ball_in_score_box(self, ball_center, hoop_bbox, velocity=None):
        """
        SIMPLE: Check if ball is in the SCORE BOX (red box)
        Only checks position - no velocity or other checks
        
        Args:
            ball_center: Current ball center (x, y)
            hoop_bbox: Hoop bounding box (x1, y1, x2, y2)
            velocity: Ignored - kept for compatibility
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
        
        # Score box: smaller horizontal margin (more precise)
        in_horizontal = abs(x - hoop_center_x) <= (hoop_width * self.score_box_horizontal_margin)
        
        # Score box: smaller vertical zone (more precise)
        upper_bound = hoop_center_y - (hoop_height * self.score_box_vertical_top_margin)
        lower_bound = y2 + (hoop_height * self.score_box_vertical_bottom_margin)
        in_vertical = upper_bound <= y <= lower_bound
        
        # SIMPLE: Just check if ball is in the box - nothing else
        return in_horizontal and in_vertical
    
    def trajectory_goes_through_score_box(self, trajectory, hoop_bbox):
        """
        Check if trajectory goes through the SCORE BOX (red box)
        This checks the entire trajectory path, not just current position
        
        Args:
            trajectory: List of (x, y, time, velocity_y) tuples
            hoop_bbox: Hoop bounding box (x1, y1, x2, y2)
        """
        if hoop_bbox is None or len(trajectory) < 1:
            return False
        
        x1, y1, x2, y2 = hoop_bbox
        hoop_center_x = (x1 + x2) / 2
        hoop_center_y = (y1 + y2) / 2
        hoop_width = x2 - x1
        hoop_height = y2 - y1
        
        # Score box boundaries (red box - smaller, precise)
        score_horizontal_margin = hoop_width * self.score_box_horizontal_margin
        score_upper = hoop_center_y - (hoop_height * self.score_box_vertical_top_margin)
        score_lower = y2 + (hoop_height * self.score_box_vertical_bottom_margin)
        
        # Check if trajectory passes through score box
        positions_in_score_box = []
        for point in trajectory:
            if len(point) >= 2:
                x, y = point[0], point[1]
                in_horizontal = abs(x - hoop_center_x) <= score_horizontal_margin
                in_vertical = score_upper <= y <= score_lower
                if in_horizontal and in_vertical:
                    positions_in_score_box.append((x, y))
        
        # Need at least 1 point in score box to confirm trajectory went through
        # Even 1 point means the ball passed through the score box
        result = len(positions_in_score_box) >= 1
        if self.debug_mode and len(positions_in_score_box) > 0:
            print(f"  [SCORE BOX CHECK] Points in score box: {len(positions_in_score_box)}, Result: {result}")
        return result
    
    def trajectory_goes_through_shot_box(self, trajectory, hoop_bbox):
        """
        Check if trajectory goes through the SHOT BOX (green box)
        This checks the entire trajectory path, not just current position
        
        Args:
            trajectory: List of (x, y, time, velocity_y) tuples
            hoop_bbox: Hoop bounding box (x1, y1, x2, y2)
        """
        if hoop_bbox is None or len(trajectory) < 1:
            return False
        
        x1, y1, x2, y2 = hoop_bbox
        hoop_center_x = (x1 + x2) / 2
        hoop_center_y = (y1 + y2) / 2
        hoop_width = x2 - x1
        hoop_height = y2 - y1
        
        # Shot box boundaries (green box - larger)
        shot_horizontal_margin = hoop_width * self.shot_box_horizontal_margin
        shot_upper = hoop_center_y - (hoop_height * self.shot_box_vertical_top_margin)
        shot_lower = y2 + (hoop_height * self.shot_box_vertical_bottom_margin)
        
        # Check if trajectory passes through shot box
        positions_in_shot_box = []
        for point in trajectory:
            if len(point) >= 2:
                x, y = point[0], point[1]
                in_horizontal = abs(x - hoop_center_x) <= shot_horizontal_margin
                in_vertical = shot_upper <= y <= shot_lower
                if in_horizontal and in_vertical:
                    positions_in_shot_box.append((x, y))
        
        # Need at least 1 point in shot box to confirm trajectory went through
        result = len(positions_in_shot_box) >= 1
        if self.debug_mode and len(positions_in_shot_box) > 0:
            print(f"  [SHOT BOX CHECK] Points in shot box: {len(positions_in_shot_box)}, Result: {result}")
        return result
    
    def is_ball_through_hoop(self, ball_center, hoop_bbox, velocity=None):
        """
        Legacy function - now uses score box
        """
        return self.is_ball_in_score_box(ball_center, hoop_bbox, velocity)
    
    def detect_miss(self, ball_center, hoop_bbox, trajectory=None):
        """
        Improved miss detection using trajectory analysis
        A miss occurs when:
        1. Ball was near hoop (approached it)
        2. Ball is now below hoop or moved away
        3. Ball never passed through hoop zone
        
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
        
        # More lenient: check if ball is below hoop (even slightly)
        below_hoop = y > hoop_bottom + (hoop_width * 0.1)  # More lenient threshold
        
        # Check horizontal alignment (ball was aimed at hoop) - more lenient
        near_hoop_horizontal = abs(x - hoop_center_x) <= (hoop_width * 0.8)
        
        # If trajectory is available, use it for better detection
        if trajectory and len(trajectory) >= 2:
            # Check if ball was near hoop recently
            recent_positions = [(pos[0], pos[1]) for pos in trajectory[-5:] if len(pos) >= 2]
            if recent_positions:
                # Check if ball was above/near hoop recently
                was_near_hoop = any(
                    self.is_point_in_bbox((pos[0], pos[1]), hoop_bbox, margin=0.8) 
                    for pos in recent_positions
                )
                
                # Ball is below hoop and was near it = miss
                if below_hoop and was_near_hoop:
                    return True
        
        # Simpler check: ball is below hoop and horizontally aligned
        return below_hoop and near_hoop_horizontal
    
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
            hoop_height = y2 - y1
            hoop_top = y1
            hoop_bottom = y2
            
            # Check if ball trajectory went through hoop zone (using same larger margins)
            positions_through_hoop = []
            for x, y in positions:
                # Check horizontal alignment (matching is_ball_through_hoop margin)
                in_horizontal = abs(x - hoop_center_x) <= (hoop_width * 0.7)  # Increased from 0.4
                # Check vertical zone (through hoop) - larger zone
                in_vertical = (hoop_top - hoop_height * 0.3) <= y <= (hoop_bottom + hoop_height * 0.5)  # Increased margins
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
    
    def process_frame(self, frame):
        """
        Process frame with YOLO detection and check for scores/misses
        
        Args:
            frame: Input frame
            
        Returns:
            Annotated frame with detections and score information
        """
        frame_h, frame_w = frame.shape[:2]
        roi_origin = (0, 0)
        inference_frame = frame

        if self.roi_enabled and self.current_roi is not None:
            x1, y1, x2, y2 = self.current_roi
            roi_w = x2 - x1
            roi_h = y2 - y1
            if roi_w >= self.roi_min_size and roi_h >= self.roi_min_size:
                roi_origin = (x1, y1)
                inference_frame = frame[y1:y2, x1:x2]
            else:
                self.current_roi = None
                self.frames_since_hoop = self.roi_reset_frames

        # Run YOLO detection (YOLO 11 API) with optimized settings
        inference_start = time.perf_counter()
        current_time = time.time()
        
        # Only detect hoop every 5 seconds (ring doesn't move, saves resources)
        should_detect_hoop = (current_time - self.last_hoop_detection_time) >= self.hoop_detection_interval
        
        # Determine which classes to detect
        if should_detect_hoop:
            # Detect both ball and hoop
            detect_classes = self.target_classes
        else:
            # Only detect ball (hoop position is cached)
            detect_classes = [0]  # Only basketball class
        
        # Use inference_mode (faster than no_grad) for maximum speed
        with self.inference_mode():
            results = self.model.predict(
                source=inference_frame,
                imgsz=self.imgsz,
                conf=0.52,  # Increased by 15% (was 0.45, now 0.52) - higher confidence for more reliable ball and hoop tracking
                device=self.device,
                half=self.use_half,
                classes=detect_classes,
                verbose=False,
                max_det=10,  # Limit max detections for faster NMS
                agnostic_nms=False,  # Class-specific NMS (faster)
                augment=False,  # Disable augmentation for speed
                retina_masks=False,  # Disable high-res masks if not needed
            )
        self.last_inference_time = time.perf_counter() - inference_start
        
        result = results[0]
        
        # Extract detections
        boxes = result.boxes
        ball_centers = []
        hoop_bboxes = []
        
        detections_to_draw = []
        
        # Process detections
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = self.class_names.get(cls_id, 'unknown')
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            bbox[[0, 2]] += roi_origin[0]
            bbox[[1, 3]] += roi_origin[1]
            confidence = float(box.conf[0])
            detections_to_draw.append((class_name, bbox, confidence))
            
            # Track ball positions (basketball or ball class)
            if class_name in ['basketball', 'ball']:
                center = self.get_bbox_center(bbox)
                ball_centers.append((center, bbox, confidence))
            
            # Track hoop position (only if we're detecting hoop this frame)
            elif class_name == 'hoop' and should_detect_hoop:
                hoop_bboxes.append(bbox)
        
        # Update hoop bbox (use first detected hoop, only if we detected it this frame)
        if hoop_bboxes:
            self.hoop_bbox = hoop_bboxes[0]
            self.last_hoop_detection_time = current_time
            self.frames_since_hoop = 0
            if self.roi_enabled:
                hx1, hy1, hx2, hy2 = self.hoop_bbox
                roi_x1 = max(int(hx1 - self.roi_margin), 0)
                roi_y1 = max(int(hy1 - self.roi_margin), 0)
                roi_x2 = min(int(hx2 + self.roi_margin), frame_w)
                roi_y2 = min(int(hy2 + self.roi_margin), frame_h)
                self.current_roi = (roi_x1, roi_y1, roi_x2, roi_y2)
        elif should_detect_hoop:
            # Only increment if we tried to detect hoop but didn't find it
            self.frames_since_hoop += 1
            if self.frames_since_hoop >= self.roi_reset_frames:
                self.current_roi = None
        
        # Process ball detections with improved trajectory tracking
        if ball_centers:
            # Filter out shadows and duplicates: prefer ball closest to last known position
            # or highest confidence if no previous position
            if self.last_ball_center is not None:
                # Calculate distance from last position for each detection
                best_ball = None
                min_distance = float('inf')
                for center, bbox, conf in ball_centers:
                    # Calculate distance from last ball center
                    dist = np.sqrt((center[0] - self.last_ball_center[0])**2 + 
                                 (center[1] - self.last_ball_center[1])**2)
                    # Also filter by reasonable size (shadows are often larger)
                    # But be more lenient - ball can appear larger when close to camera/ring
                    bbox_w = bbox[2] - bbox[0]
                    bbox_h = bbox[3] - bbox[1]
                    bbox_area = bbox_w * bbox_h
                    # Reasonable ball size (increased threshold - ball can be larger near ring)
                    # Also allow larger distance when ball is moving fast (near ring)
                    max_distance = 200 if self.shot_state == 'shooting' else 100  # Allow larger jumps during shots
                    if bbox_area < 100000:  # Increased from 50000 - ball can appear larger near ring
                        if dist < min_distance and dist < max_distance:
                            min_distance = dist
                            best_ball = (center, bbox, conf)
                if best_ball:
                    ball_center, ball_bbox, ball_confidence = best_ball
                else:
                    # Fallback: use highest confidence, but be more lenient with distance
                    # During active shots, accept larger distances
                    if self.shot_state == 'shooting':
                        # During shot, accept any detection with reasonable confidence
                        ball_centers.sort(key=lambda x: x[2], reverse=True)
                        ball_center, ball_bbox, ball_confidence = ball_centers[0]
                    else:
                        # Normal fallback
                        ball_centers.sort(key=lambda x: x[2], reverse=True)
                        ball_center, ball_bbox, ball_confidence = ball_centers[0]
            else:
                # No previous position: use highest confidence, but filter by size (more lenient)
                ball_centers_filtered = []
                for center, bbox, conf in ball_centers:
                    bbox_w = bbox[2] - bbox[0]
                    bbox_h = bbox[3] - bbox[1]
                    bbox_area = bbox_w * bbox_h
                    if bbox_area < 100000:  # Increased from 50000 - ball can appear larger near ring
                        ball_centers_filtered.append((center, bbox, conf))
                if ball_centers_filtered:
                    ball_centers_filtered.sort(key=lambda x: x[2], reverse=True)
                    ball_center, ball_bbox, ball_confidence = ball_centers_filtered[0]
                else:
                    # Fallback if all filtered out - use highest confidence anyway
                    ball_centers.sort(key=lambda x: x[2], reverse=True)
                    ball_center, ball_bbox, ball_confidence = ball_centers[0]
            
            # Calculate velocity if we have previous position
            velocity = None
            if self.last_ball_center is not None:
                # Estimate previous time based on target FPS
                prev_time = current_time - (1.0 / self.target_fps)
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
            
            # Update shot state machine - Two-box system: Shot box (larger) and Score box (smaller)
            # TRAJECTORY-BASED SYSTEM: Everything based on trajectory line (orange/yellow line)
            if self.hoop_bbox is not None:
                # Always check trajectory first - this is the ONLY source of truth
                # Don't use current ball position, only trajectory
                trajectory_through_score = self.trajectory_goes_through_score_box(self.ball_trajectory, self.hoop_bbox)
                trajectory_through_shot = self.trajectory_goes_through_shot_box(self.ball_trajectory, self.hoop_bbox)
                
                if self.debug_mode and len(self.ball_trajectory) > 0:
                    print(f"\n[DEBUG] State: {self.shot_state}, Trajectory points: {len(self.ball_trajectory)}, Through Shot Box: {trajectory_through_shot}, Through Score Box: {trajectory_through_score}, Result registered: {self.shot_result_registered}")
                
                if self.shot_state == 'idle':
                    # Detect shot: trajectory goes through green box (shot box)
                    if trajectory_through_shot:
                        # Shot detected via trajectory! Start tracking this shot
                        self.shot_state = 'shooting'
                        self.ball_in_hoop_zone = True
                        self.ball_entry_time = current_time
                        self.ball_scored_in_current_entry = False
                        self.shot_result_registered = False  # Reset for new shot
                        self.trajectory_length_at_result = None  # Reset trajectory tracking
                        self.current_shot_id += 1  # Increment shot ID for new shot
                        if self.debug_mode:
                            print(f"  -> Shot detected via trajectory! (trajectory through shot box, Shot ID: {self.current_shot_id})")
                
                elif self.shot_state == 'shooting':
                    # CRITICAL: Only allow ONE result per trajectory
                    # Track trajectory length to prevent duplicate results for same trajectory
                    current_trajectory_length = len(self.ball_trajectory)
                    
                    # If we already registered a result for this trajectory length, skip
                    if self.trajectory_length_at_result is not None and current_trajectory_length <= self.trajectory_length_at_result:
                        # This trajectory segment was already processed, skip
                        if self.debug_mode:
                            print(f"  -> Trajectory already processed (length {current_trajectory_length} <= {self.trajectory_length_at_result}), skipping")
                    elif not self.shot_result_registered:
                        # Check score box - this is the most important check
                        if trajectory_through_score:
                            # Trajectory went through score box = SCORE!
                            if current_time - self.last_score_time > self.score_cooldown:
                                self.score += 1
                                self.last_score_time = current_time
                                self.ball_scored_in_current_entry = True
                                self.shot_result_registered = True
                                self.trajectory_length_at_result = current_trajectory_length  # Mark this trajectory as processed
                                self.shot_state = 'scored'
                                print(f"🎯 SCORE! (Trajectory through score box) Total: {self.score} (Shot ID: {self.current_shot_id})")
                                if self.debug_mode:
                                    print(f"  -> SCORE registered via trajectory (Shot ID: {self.current_shot_id}, Trajectory length: {current_trajectory_length})")
                        # Only check for miss if trajectory is no longer in shot box AND we haven't scored
                        elif not trajectory_through_shot:
                            # Shot finished - trajectory left shot box
                            # Final check: make absolutely sure trajectory didn't go through score box
                            final_score_check = self.trajectory_goes_through_score_box(self.ball_trajectory, self.hoop_bbox)
                            if final_score_check:
                                # Trajectory went through score box = SCORE!
                                if current_time - self.last_score_time > self.score_cooldown:
                                    self.score += 1
                                    self.last_score_time = current_time
                                    self.ball_scored_in_current_entry = True
                                    self.shot_result_registered = True
                                    self.trajectory_length_at_result = current_trajectory_length  # Mark this trajectory as processed
                                    self.shot_state = 'scored'
                                    print(f"🎯 SCORE! (Final check - trajectory through score box) Total: {self.score} (Shot ID: {self.current_shot_id})")
                                    if self.debug_mode:
                                        print(f"  -> SCORE registered via final trajectory check (Shot ID: {self.current_shot_id}, Trajectory length: {current_trajectory_length})")
                            else:
                                # Trajectory went through shot box but NOT score box = MISS
                                if current_time - self.last_miss_time > self.miss_cooldown:
                                    self.record_miss(event_time=current_time)
                                    self.shot_result_registered = True
                                    self.trajectory_length_at_result = current_trajectory_length  # Mark this trajectory as processed
                                    self.shot_state = 'missed'
                                    print(f"❌ MISS! (Trajectory through shot box, not score box) Total misses: {self.misses} (Shot ID: {self.current_shot_id})")
                                    if self.debug_mode:
                                        print(f"  -> MISS registered via trajectory (Shot ID: {self.current_shot_id}, Trajectory length: {current_trajectory_length})")
                    else:
                        # Result already registered for this shot, reset
                        self.shot_state = 'idle'
                        if self.debug_mode:
                            print(f"  -> State reset to IDLE (result registered)")
                
                elif self.shot_state == 'scored':
                    # After scoring, reset quickly to allow tracking to continue
                    # Reset after short cooldown OR when trajectory no longer in shot box
                    # IMPORTANT: Ball tracking continues regardless of state!
                    reset_after_score = (current_time - self.last_score_time > 1.0) or (not trajectory_through_shot and current_time - self.last_score_time > 0.3)
                    if reset_after_score:
                        self.shot_state = 'idle'
                        self.ball_in_hoop_zone = False
                        self.shot_result_registered = False  # Reset for next shot
                        self.trajectory_length_at_result = None  # Reset trajectory tracking
                        if self.debug_mode:
                            print(f"  -> State reset to IDLE after score (ready for next shot)")
                    # Allow new shot detection even in 'scored' state if trajectory comes back in shot box
                    # This ensures continuous tracking
                    if trajectory_through_shot and (current_time - self.last_score_time > 0.5):
                        # Trajectory is back in shot box - prepare for potential new shot
                        if self.debug_mode:
                            print(f"  -> Trajectory back in shot box after score, preparing for new shot")
                
                elif self.shot_state == 'missed':
                    # After miss, reset quickly to allow tracking to continue
                    reset_after_miss = (current_time - self.last_miss_time > 1.0) or (not trajectory_through_shot and current_time - self.last_miss_time > 0.3)
                    if reset_after_miss:
                        self.shot_state = 'idle'
                        self.ball_in_hoop_zone = False
                        self.shot_result_registered = False  # Reset for next shot
                        self.trajectory_length_at_result = None  # Reset trajectory tracking
                        if self.debug_mode:
                            print(f"  -> State reset to IDLE after miss (ready for next shot)")
                
                # Fallback: ALWAYS check trajectory for score box - catch any missed scores
                # But only if this trajectory hasn't been processed yet
                if self.shot_state != 'scored' and not self.shot_result_registered:
                    current_trajectory_length = len(self.ball_trajectory)
                    # Only check if this trajectory segment hasn't been processed
                    if self.trajectory_length_at_result is None or current_trajectory_length > self.trajectory_length_at_result:
                        trajectory_through_score_fallback = self.trajectory_goes_through_score_box(self.ball_trajectory, self.hoop_bbox)
                        if trajectory_through_score_fallback and current_time - self.last_score_time > self.score_cooldown:
                            self.score += 1
                            self.last_score_time = current_time
                            self.ball_scored_in_current_entry = True
                            self.shot_result_registered = True
                            self.trajectory_length_at_result = current_trajectory_length  # Mark as processed
                            self.shot_state = 'scored'
                            print(f"🎯 SCORE! (fallback check - trajectory through score box) Total: {self.score} (Shot ID: {self.current_shot_id})")
                            if self.debug_mode:
                                print(f"  -> Fallback score detected via trajectory (Shot ID: {self.current_shot_id}, Trajectory length: {current_trajectory_length})")
            
            self.last_ball_center = ball_center
            self.last_ball_velocity = velocity
            self.frames_without_ball = 0  # Reset counter when ball is detected
        else:
            # No ball detected - increment counter
            self.frames_without_ball += 1
            
            # Reset tracking if ball not detected for too long
            # BUT: Don't reset during active shots - ball might be temporarily occluded
            if self.frames_without_ball >= self.max_prediction_frames:
                # Only reset if not in active shot state
                if self.shot_state == 'idle':
                    # No active shot, safe to reset
                    self.last_ball_center = None
                    self.last_ball_velocity = None
                elif self.shot_state in ['shooting']:
                    # During active shot, keep tracking even if ball temporarily not detected
                    # Ball might be occluded by ring or moving fast
                    # Only reset if we've been without ball for much longer
                    if self.frames_without_ball >= self.max_prediction_frames * 2:
                        # Ball really gone, finalize shot as miss if not scored
                        if not self.shot_result_registered:
                            if current_time - self.last_miss_time > self.miss_cooldown:
                                self.record_miss(event_time=current_time)
                                self.shot_result_registered = True
                                self.shot_state = 'missed'
                                print(f"❌ MISS! (ball lost) Total misses: {self.misses}")
                        self.last_ball_center = None
                        self.last_ball_velocity = None
                else:
                    # Scored or missed state - can reset
                    self.last_ball_center = None
                    self.last_ball_velocity = None
        
        # Manual drawing for better performance
        annotated_frame = frame.copy()
        
        # Draw trajectory path if available (limit to last 20 points for performance)
        if len(self.ball_trajectory) >= 2:
            # Only draw last 20 points for better FPS
            recent_trajectory = self.ball_trajectory[-20:] if len(self.ball_trajectory) > 20 else self.ball_trajectory
            trajectory_points = [(int(x), int(y)) for x, y, _, _ in recent_trajectory]
            for i in range(len(trajectory_points) - 1):
                # Draw trajectory line with fading color (recent = brighter)
                alpha = i / len(trajectory_points)
                color_intensity = int(255 * (1 - alpha * 0.7))
                cv2.line(annotated_frame, trajectory_points[i], trajectory_points[i+1], 
                        (0, color_intensity, 255), 2)
        
        # Draw detections (basketball tracking - bounding boxes only)
        for class_name, bbox, confidence in detections_to_draw:
            color = (0, 165, 255) if class_name in ['basketball', 'ball'] else (0, 255, 255)
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Make hoop box (blue/cyan) larger for better visibility and detection
            if class_name == 'hoop':
                # Expand hoop box by 20% on all sides for better detection zone
                width = x2 - x1
                height = y2 - y1
                margin = 0.2  # 20% expansion
                x1 = int(x1 - width * margin)
                y1 = int(y1 - height * margin)
                x2 = int(x2 + width * margin)
                y2 = int(y2 + height * margin)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw TWO-BOX SYSTEM: Shot box (larger) and Score box (smaller)
        if self.hoop_bbox is not None:
            x1, y1, x2, y2 = self.hoop_bbox
            width = x2 - x1
            height = y2 - y1
            hoop_center_x = (x1 + x2) / 2
            hoop_center_y = (y1 + y2) / 2
            
            # SHOT BOX (larger, outer box) - Green color
            shot_horizontal_margin = width * self.shot_box_horizontal_margin
            shot_x1 = int(hoop_center_x - shot_horizontal_margin)
            shot_x2 = int(hoop_center_x + shot_horizontal_margin)
            shot_y1 = int(hoop_center_y - height * self.shot_box_vertical_top_margin)
            shot_y2 = int(y2 + height * self.shot_box_vertical_bottom_margin)
            
            # Draw shot box in GREEN (larger area for shot attempts)
            shot_box_color = (0, 255, 0)  # Green
            cv2.rectangle(annotated_frame, (shot_x1, shot_y1), (shot_x2, shot_y2), shot_box_color, 2)
            
            # SCORE BOX (smaller, inner box) - Red/Blue color
            score_horizontal_margin = width * self.score_box_horizontal_margin
            score_x1 = int(hoop_center_x - score_horizontal_margin)
            score_x2 = int(hoop_center_x + score_horizontal_margin)
            score_y1 = int(hoop_center_y - height * self.score_box_vertical_top_margin)
            score_y2 = int(y2 + height * self.score_box_vertical_bottom_margin)
            
            # Draw score box in RED (smaller, precise area for scores)
            score_box_color = (0, 0, 255)  # Red
            cv2.rectangle(annotated_frame, (score_x1, score_y1), (score_x2, score_y2), score_box_color, 2)

        return annotated_frame
    
    def record_miss(self, event_time=None, force=False):
        """
        Increment miss counter with cooldown handling

        Args:
            event_time (float): Timestamp to compare against cooldown
            force (bool): When True, ignore cooldown (for manual input)
        """
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
        print("Live Camera Detection Started!")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit/Close")
        print("  's' - Save current frame")
        print("  'm' - Mark manual miss")
        print("="*60 + "\n")
        
        # High-precision FPS tracking with smoothing
        fps = 0.0
        prev_time = time.perf_counter()
        smooth_alpha = 0.9  # higher = smoother, slower to respond
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
                
                # Process frame with YOLO
                annotated_frame = self.process_frame(frame)
                
                # Calculate FPS (exponential moving average per frame)
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
                cv2.imshow('YOLO Live Detection - Press Q to quit', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_snapshot_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"✓ Frame saved as {filename}")
                elif key == ord('m'):
                    self.record_miss(force=True)
                
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
    """Main function to run the live camera detection"""
    print("\n" + "="*60)
    print("YOLO 11 Live Camera Detection Application")
    print("="*60)
    print("Using your trained YOLO 11 model (best.pt)")
    print("="*60)
    
    try:
        # Initialize detection system
        detector = LiveCameraDetection('best.pt')
        
        # Run detection
        detector.run_detection()
        
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

