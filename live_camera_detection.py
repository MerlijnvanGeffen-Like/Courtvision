import cv2
from ultralytics import YOLO
import time
import torch

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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_half = torch.cuda.is_available()
        self.imgsz = 384  # smaller input for higher FPS
        self.load_model()
        
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
            
            print(f"✓ Model loaded successfully!")
            print(f"✓ Detecting: {list(self.class_names.values())}")
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
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            
            print(f"Camera {camera_index} initialized successfully!")
            return True
        except Exception as e:
            print(f"Error setting up camera: {e}")
            return False
    
    def process_frame(self, frame):
        """
        Process frame with YOLO detection
        
        Args:
            frame: Input frame
            
        Returns:
            Annotated frame with detections
        """
        # Run YOLO detection (v8 API) with perf-friendly settings
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=0.25,
            device=self.device,
            half=self.use_half,
            verbose=False
        )
        
        # Use YOLO's built-in visualization
        annotated_frame = results[0].plot()
        
        return annotated_frame
    
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
        print("="*60 + "\n")
        
        # High-precision FPS tracking with smoothing
        fps = 0.0
        prev_time = time.perf_counter()
        smooth_alpha = 0.9  # higher = smoother, slower to respond
        
        try:
            while True:
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
                
                # Display FPS on frame
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

