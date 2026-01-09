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
        self.imgsz = 288  # balance between accuracy and FPS
        self.capture_width = 480
        self.capture_height = 270
        self.target_fps = 30
        self.roi_enabled = True
        self.roi_margin = 140
        self.roi_min_size = 200
        self.roi_reset_frames = 40
        self.frames_since_hoop = self.roi_reset_frames
        self.current_roi = None  # (x1, y1, x2, y2)
        
        # Detection preferences (basketball + hoop only)
        self.target_classes = [0, 3]

        # Scoring system
        self.score = 0
        self.misses = 0
        self.ball_positions = []  # Track ball positions over time
        self.hoop_bbox = None  # Current hoop bounding box
        self.last_score_time = 0  # Prevent duplicate scoring
        self.last_miss_time = 0  # Prevent duplicate miss detection
        self.ball_in_hoop_zone = False  # Track if ball is near hoop
        self.ball_entry_time = 0
        self.ball_scored_in_current_entry = False
        self.last_ball_center = None
        self.score_cooldown = 1.0  # Seconds between score detections
        self.miss_cooldown = 2.0  # Seconds between miss detections
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
        """Load the trained YOLO model"""
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
            try:
                self.model.fuse()
            except Exception:
                pass
            if self.use_half:
                try:
                    self.model.model.half()
                except Exception:
                    self.use_half = False
            
            # Extra backend perf tweaks
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
                cv2.setNumThreads(0)
            except Exception:
                pass

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
    
    def check_ball_through_hoop(self, ball_center, hoop_bbox):
        """
        Check if ball has passed through the hoop
        
        Args:
            ball_center: Current ball center (x, y)
            hoop_bbox: Hoop bounding box (x1, y1, x2, y2)
        """
        if hoop_bbox is None:
            return False
        
        # Check if ball center is within hoop bounds (with some margin)
        # For a score, ball should be in the upper-middle part of hoop
        x, y = ball_center
        x1, y1, x2, y2 = hoop_bbox
        
        # Calculate hoop center and dimensions
        hoop_center_x = (x1 + x2) / 2
        hoop_center_y = (y1 + y2) / 2
        hoop_width = x2 - x1
        hoop_height = y2 - y1
        
        # Require ball to be reasonably centered and below hoop midpoint
        horizontal_margin = 0.35  # 35% of width on each side
        lower_zone_start = hoop_center_y  # half height
        lower_zone_end = y2 + (hoop_height * 0.15)  # allow small margin below hoop
        
        in_horizontal = abs(x - hoop_center_x) <= (hoop_width * horizontal_margin)
        in_vertical = lower_zone_start <= y <= lower_zone_end
        
        return in_horizontal and in_vertical
    
    def detect_miss(self, ball_center, hoop_bbox):
        """
        Detect if ball missed the hoop (ball was near hoop but didn't go through)
        
        Args:
            ball_center: Current ball center (x, y)
            hoop_bbox: Hoop bounding box (x1, y1, x2, y2)
        """
        if hoop_bbox is None:
            return False
        
        x, y = ball_center
        x1, y1, x2, y2 = hoop_bbox
        
        # Ball is near hoop but below it (missed)
        hoop_bottom = y2
        hoop_center_x = (x1 + x2) / 2
        hoop_width = x2 - x1
        
        # Check if ball is below hoop and horizontally aligned
        near_hoop_horizontal = abs(x - hoop_center_x) <= (hoop_width * 0.6)
        below_hoop = y > hoop_bottom + 20  # 20 pixels below hoop
        
        return near_hoop_horizontal and below_hoop
    
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

        # Run YOLO detection (v8 API) with perf-friendly settings
        inference_start = time.perf_counter()
        results = self.model.predict(
            source=inference_frame,
            imgsz=self.imgsz,
            conf=0.75,
            device=self.device,
            half=self.use_half,
            classes=self.target_classes,
            verbose=False
        )
        self.last_inference_time = time.perf_counter() - inference_start
        
        result = results[0]
        current_time = time.time()
        
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
                ball_centers.append((center, bbox))
            
            # Track hoop position
            elif class_name == 'hoop':
                hoop_bboxes.append(bbox)
        
        # Update hoop bbox (use first detected hoop)
        if hoop_bboxes:
            self.hoop_bbox = hoop_bboxes[0]
            self.frames_since_hoop = 0
            if self.roi_enabled:
                hx1, hy1, hx2, hy2 = self.hoop_bbox
                roi_x1 = max(int(hx1 - self.roi_margin), 0)
                roi_y1 = max(int(hy1 - self.roi_margin), 0)
                roi_x2 = min(int(hx2 + self.roi_margin), frame_w)
                roi_y2 = min(int(hy2 + self.roi_margin), frame_h)
                self.current_roi = (roi_x1, roi_y1, roi_x2, roi_y2)
        else:
            self.frames_since_hoop += 1
            if self.frames_since_hoop >= self.roi_reset_frames:
                self.current_roi = None
        
        # Process ball detections
        if ball_centers:
            # Use the ball with highest confidence or first one
            ball_center, ball_bbox = ball_centers[0]
            self.ball_positions.append((ball_center, current_time))
            
            # Keep only recent positions (last 1 second)
            self.ball_positions = [(pos, t) for pos, t in self.ball_positions if current_time - t < 1.0]
            
            # Check for score
            if self.hoop_bbox is not None:
                ball_near_hoop = self.is_point_in_bbox(ball_center, self.hoop_bbox, margin=0.25)
                
                if ball_near_hoop and not self.ball_in_hoop_zone:
                    self.ball_in_hoop_zone = True
                    self.ball_entry_time = current_time
                    self.ball_scored_in_current_entry = False
                
                if ball_near_hoop and self.check_ball_through_hoop(ball_center, self.hoop_bbox):
                    downward_motion = True
                    if self.last_ball_center is not None:
                        downward_motion = ball_center[1] >= self.last_ball_center[1] - 2
                    
                    if downward_motion and current_time - self.last_score_time > self.score_cooldown:
                        self.score += 1
                        self.last_score_time = current_time
                        self.ball_scored_in_current_entry = True
                        print(f"SCORE! Total: {self.score}")
                elif self.ball_in_hoop_zone and not ball_near_hoop:
                    if not self.ball_scored_in_current_entry:
                        self.record_miss(event_time=current_time)
                    self.ball_in_hoop_zone = False
                
                # Fallback: if ball drops below hoop while still tracked but no score
                if (self.ball_in_hoop_zone and not self.ball_scored_in_current_entry and 
                        self.detect_miss(ball_center, self.hoop_bbox)):
                    self.record_miss(event_time=current_time)
                    self.ball_in_hoop_zone = False
            
            self.last_ball_center = ball_center
        else:
            self.last_ball_center = None
        
        # Manual drawing for better performance
        annotated_frame = frame.copy()
        for class_name, bbox, confidence in detections_to_draw:
            color = (0, 165, 255) if class_name in ['basketball', 'ball'] else (0, 255, 255)
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw score and misses on frame
        score_text = f"Score: {self.score}"
        miss_text = f"Misses: {self.misses}"
        
        # Draw score (green)
        cv2.putText(annotated_frame, score_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Draw misses (red)
        cv2.putText(annotated_frame, miss_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
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
                
                # Display FPS and perf stats on frame
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(annotated_frame, f"Infer: {self.last_inference_time*1000:.1f} ms",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
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
                
                # Pace loop to target FPS if processing was faster
                loop_time = time.perf_counter() - frame_start
                self.last_frame_time = loop_time
                if loop_time < target_frame_time:
                    time.sleep(target_frame_time - loop_time)
                
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
    print("🚀 YOLO Live Camera Detection Application")
    print("="*60)
    print("Using your trained YOLO model (best.pt)")
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

