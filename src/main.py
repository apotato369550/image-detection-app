#!/usr/bin/env python3
"""
Computer Vision Application Main Entry Point

This module serves as the main entry point for the complete computer vision application.
It integrates model downloading, camera capture, object detection, and visualization
into a cohesive real-time object detection system.
"""

import cv2
import sys
import logging
import time
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComputerVisionApp:
    """Main application class for the complete computer vision system."""

    def __init__(self,
                 camera_id: int = 0,
                 model_name: str = "ssd_mobilenet_v2",
                 confidence_threshold: float = 0.5,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30):
        """
        Initialize the computer vision application.

        Args:
            camera_id: Camera device index
            model_name: Name of the model to use
            confidence_threshold: Detection confidence threshold
            width: Camera frame width
            height: Camera frame height
            fps: Camera frame rate
        """
        self.camera_id = camera_id
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.width = width
        self.height = height
        self.fps = fps

        # Core components (initialized lazily)
        self.model_manager = None
        self.camera = None
        self.detector = None
        self.visualizer = None

        # Application state
        self.running = False
        self.frame_count = 0
        self.start_time = 0.0

        # Performance tracking
        self.total_inference_time = 0.0

    def initialize_model_manager(self) -> bool:
        """Initialize the model manager and download models."""
        try:
            from models.model_manager import ModelManager

            logger.info("Initializing model manager...")
            self.model_manager = ModelManager()

            # Check if model exists, download if needed
            if not self.model_manager.model_exists(self.model_name):
                logger.info(f"Downloading model: {self.model_name}")
                model_path = self.model_manager.download_model(self.model_name)
                if not model_path:
                    logger.error(f"Failed to download model: {self.model_name}")
                    return False
                logger.info(f"Model downloaded to: {model_path}")
            else:
                logger.info(f"Model {self.model_name} already exists")

            return True

        except ImportError as e:
            logger.error(f"Failed to import model manager: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing model manager: {str(e)}")
            return False

    def initialize_detector(self) -> bool:
        """Initialize the object detector."""
        try:
            from models.detector import ObjectDetector

            logger.info("Initializing object detector...")

            # Get model path
            if not self.model_manager:
                logger.error("Model manager not initialized")
                return False

            model_info = self.model_manager.get_model_info(self.model_name)
            if not model_info:
                logger.error(f"Model info not found for: {self.model_name}")
                return False

            # Fix: tf.saved_model.load() expects the directory containing the SavedModel, not the .pb file
            model_file = Path("models")

            # Initialize detector
            self.detector = ObjectDetector(confidence_threshold=self.confidence_threshold)
            success = self.detector.load_model(str(model_file))

            if success:
                logger.info("Object detector initialized successfully")
                return True
            else:
                logger.error("Failed to load model into detector")
                return False

        except ImportError as e:
            logger.error(f"Failed to import detector: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing detector: {str(e)}")
            return False

    def initialize_camera(self) -> bool:
        """Initialize the camera handler."""
        try:
            from utils.camera import CameraHandler

            logger.info(f"Initializing camera {self.camera_id}...")
            self.camera = CameraHandler(
                camera_id=self.camera_id,
                width=self.width,
                height=self.height,
                fps=self.fps
            )

            if self.camera.initialize():
                logger.info("Camera initialized successfully")
                return True
            else:
                logger.error("Failed to initialize camera")
                return False

        except ImportError as e:
            logger.error(f"Failed to import camera handler: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            return False

    def initialize_visualizer(self) -> bool:
        """Initialize the detection visualizer."""
        try:
            from utils.visualizer import DetectionVisualizer

            logger.info("Initializing visualizer...")
            self.visualizer = DetectionVisualizer(
                box_thickness=2,
                text_scale=0.6,
                text_thickness=2
            )

            logger.info("Visualizer initialized successfully")
            return True

        except ImportError as e:
            logger.error(f"Failed to import visualizer: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing visualizer: {str(e)}")
            return False

    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Starting Computer Vision Application initialization...")

        # Initialize components in order
        if not self.initialize_model_manager():
            return False

        if not self.initialize_detector():
            return False

        if not self.initialize_camera():
            return False

        if not self.initialize_visualizer():
            return False

        self.start_time = time.time()
        logger.info("All components initialized successfully")
        return True

    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        """Process a single frame through the detection pipeline."""
        if not self.detector or not self.visualizer:
            logger.error("Detector or visualizer not initialized")
            return frame

        try:
            # Run object detection
            detection_result = self.detector.detect(frame)

            # Update performance tracking
            self.total_inference_time += detection_result.inference_time

            # Debug: Log detection results
            num_detections = len(detection_result.detections)
            logger.info(f"Frame processed: {num_detections} objects detected")

            if num_detections > 0:
                logger.info(f"Top detection: {detection_result.detections[0].class_name} ({detection_result.detections[0].confidence:.2f})")

            # Draw detections on frame
            result_frame = self.visualizer.draw_detections(
                frame, detection_result.detections
            )

            return result_frame

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame

    def add_performance_overlay(self, frame: cv2.Mat) -> cv2.Mat:
        """Add performance information overlay to frame."""
        try:
            # Calculate FPS
            elapsed_time = time.time() - self.start_time if self.start_time > 0 else 0

            if elapsed_time > 0 and self.frame_count > 0:
                fps = self.frame_count / elapsed_time
                # Fix division by zero: handle case where total_inference_time is 0
                if self.total_inference_time > 0:
                    inference_fps = 1.0 / (self.total_inference_time / max(self.frame_count, 1))
                else:
                    inference_fps = 0.0

                # Add text overlay
                info_text = f"FPS: {fps:.1f} | Inference: {inference_fps:.1f} FPS"

                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                           cv2.LINE_AA)
            else:
                # Show placeholder when no timing data is available
                info_text = "FPS: -- | Inference: -- FPS"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                           cv2.LINE_AA)

            return frame

        except Exception as e:
            logger.error(f"Error adding performance overlay: {str(e)}")
            return frame

    def run(self) -> None:
        """Main application loop."""
        logger.info("Starting Computer Vision Application...")

        if not self.initialize():
            logger.error("Failed to initialize application. Exiting.")
            sys.exit(1)

        self.running = True
        logger.info("Application started. Press 'q' to quit, 's' to save frame")

        try:
            # Use camera context manager for automatic cleanup
            with self.camera:
                for frame in self.camera.get_frame_generator():
                    if not self.running:
                        break

                    # Process frame through detection pipeline
                    processed_frame = self.process_frame(frame)

                    # Add performance overlay
                    final_frame = self.add_performance_overlay(processed_frame)

                    # Display the frame
                    cv2.imshow('Computer Vision App - Object Detection', final_frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        logger.info("Quit key pressed")
                        break
                    elif key == ord('s'):
                        # Save current frame
                        timestamp = int(time.time())
                        filename = f"detection_result_{timestamp}.jpg"
                        cv2.imwrite(filename, final_frame)
                        logger.info(f"Frame saved as: {filename}")
                    elif key == ord('p'):
                        # Print performance stats
                        if self.detector:
                            stats = self.detector.get_performance_stats()
                            logger.info(f"Performance stats: {stats}")

                    self.frame_count += 1

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources...")

        self.running = False

        # Components cleanup is handled by context managers and individual cleanup methods
        if self.camera:
            self.camera.disconnect()

        cv2.destroyAllWindows()
        logger.info("Application closed")

    def get_performance_summary(self) -> dict:
        """Get comprehensive performance summary."""
        elapsed_time = time.time() - self.start_time if self.start_time > 0 else 0

        summary = {
            "total_frames": self.frame_count,
            "total_runtime": elapsed_time,
            "average_fps": self.frame_count / elapsed_time if elapsed_time > 0 else 0,
            "total_inference_time": self.total_inference_time,
            "average_inference_time": self.total_inference_time / max(self.frame_count, 1)
        }

        if self.detector:
            detector_stats = self.detector.get_performance_stats()
            summary.update(detector_stats)

        return summary


def setup_logging(level: str = "INFO") -> None:
    """Setup application logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main() -> None:
    """Main entry point function."""
    # Setup logging
    setup_logging("INFO")

    # Create and run application
    app = ComputerVisionApp(
        camera_id=0,
        model_name="ssd_mobilenet_v2",
        confidence_threshold=0.3,  # Back to reasonable threshold now that input format is fixed
        width=640,
        height=480,
        fps=30
    )

    try:
        app.run()

        # Print final performance summary
        summary = app.get_performance_summary()
        logger.info("Performance Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()