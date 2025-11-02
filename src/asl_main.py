#!/usr/bin/env python3
"""
ASL Hand Sign Detection Application - Main Entry Point

This module serves as the main entry point for the ASL hand sign detection application.
It integrates MediaPipe hand detection, ASL sign classification, camera capture,
and visualization into a cohesive real-time ASL recognition system.

Detects American Sign Language alphabet letters (A-Z) using hand pose estimation.
Note: Letters J and Z require motion and are not reliably detected with static poses.
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


class ASLHandSignApp:
    """Main application class for the ASL hand sign detection system."""

    def __init__(self,
                 camera_id: int = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 draw_landmarks: bool = True):
        """
        Initialize the ASL hand sign detection application.

        Args:
            camera_id: Camera device index
            width: Camera frame width
            height: Camera frame height
            fps: Camera frame rate
            draw_landmarks: Whether to draw hand landmarks
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.draw_landmarks = draw_landmarks

        # Core components (initialized lazily)
        self.camera = None
        self.asl_detector = None
        self.asl_visualizer = None

        # Application state
        self.running = False
        self.frame_count = 0
        self.start_time = 0.0

        # Performance tracking
        self.total_inference_time = 0.0
        self.detected_signs = []

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

    def initialize_asl_detector(self) -> bool:
        """Initialize the ASL hand sign detector."""
        try:
            from models.asl_detector import ASLHandDetector

            logger.info("Initializing ASL hand sign detector...")
            self.asl_detector = ASLHandDetector(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                max_num_hands=2
            )

            if self.asl_detector.load_model():
                logger.info("ASL hand sign detector initialized successfully")
                logger.info(f"Can recognize {len(self.asl_detector.asl_letters)} ASL letters")
                return True
            else:
                logger.error("Failed to load ASL detector model")
                return False

        except ImportError as e:
            logger.error(f"Failed to import ASL detector: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing ASL detector: {str(e)}")
            return False

    def initialize_visualizer(self) -> bool:
        """Initialize the ASL detection visualizer."""
        try:
            from utils.asl_visualizer import ASLVisualizer

            logger.info("Initializing ASL visualizer...")
            self.asl_visualizer = ASLVisualizer(
                box_thickness=2,
                text_scale=0.7,
                text_thickness=2,
                landmark_size=3
            )

            logger.info("ASL visualizer initialized successfully")
            return True

        except ImportError as e:
            logger.error(f"Failed to import visualizer: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing visualizer: {str(e)}")
            return False

    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Starting ASL Hand Sign Detection Application initialization...")

        # Initialize components in order
        if not self.initialize_camera():
            return False

        if not self.initialize_asl_detector():
            return False

        if not self.initialize_visualizer():
            return False

        self.start_time = time.time()
        logger.info("All components initialized successfully")
        return True

    def process_frame(self, frame: cv2.Mat) -> tuple:
        """Process a single frame through the detection pipeline.

        Returns:
            tuple: (result_frame, detection_result) for access to current detections
        """
        if not self.asl_detector or not self.asl_visualizer:
            logger.error("Detector or visualizer not initialized")
            return frame, None

        try:
            # Run ASL hand sign detection
            detection_result = self.asl_detector.detect(frame)

            # Update performance tracking
            self.total_inference_time += detection_result.inference_time

            # Debug: Log detection results
            num_detections = len(detection_result.detections)

            if num_detections > 0:
                # Update detected signs history
                for detection in detection_result.detections:
                    self.detected_signs.append(detection.letter)
                    # Keep only last 20 signs
                    if len(self.detected_signs) > 20:
                        self.detected_signs = self.detected_signs[-20:]

                logger.debug(f"Frame {self.frame_count}: Detected {num_detections} hand(s)")
                for i, det in enumerate(detection_result.detections):
                    logger.debug(f"  Hand {i+1}: {det.letter} ({det.confidence:.2f}) - {det.handedness}")

            # Draw detections on frame
            result_frame = self.asl_visualizer.draw_asl_detections(
                frame, detection_result.detections, self.draw_landmarks
            )

            # Add legend showing detected signs
            recent_letters = list(dict.fromkeys(self.detected_signs[-5:]))  # Last 5 unique
            if recent_letters:
                result_frame = self.asl_visualizer.draw_asl_legend(result_frame, recent_letters)

            return result_frame, detection_result

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, None

    def add_performance_overlay(self, frame: cv2.Mat, detection_result=None) -> cv2.Mat:
        """Add performance information overlay to frame.

        Args:
            frame: Input frame
            detection_result: Current frame's detection result to get accurate hand count
        """
        try:
            # Calculate FPS
            elapsed_time = time.time() - self.start_time if self.start_time > 0 else 0

            if elapsed_time > 0 and self.frame_count > 0:
                fps = self.frame_count / elapsed_time
            else:
                fps = 0.0

            # Get current hand count from detection result
            current_hands = len(detection_result.detections) if detection_result else 0

            # Add info overlay
            if self.asl_visualizer:
                frame = self.asl_visualizer.draw_info_overlay(
                    frame, fps, current_hands
                )

            return frame

        except Exception as e:
            logger.error(f"Error adding performance overlay: {str(e)}")
            return frame

    def run(self) -> None:
        """Main application loop."""
        logger.info("Starting ASL Hand Sign Detection Application...")

        if not self.initialize():
            logger.error("Failed to initialize application. Exiting.")
            sys.exit(1)

        self.running = True
        logger.info("ASL Detection started. Press 'q' to quit, 's' to save frame, 'l' to toggle landmarks")

        try:
            # Use camera context manager for automatic cleanup
            with self.camera:
                for frame in self.camera.get_frame_generator():
                    if not self.running:
                        break

                    # Process frame through detection pipeline
                    processed_frame, detection_result = self.process_frame(frame)

                    # Add performance overlay
                    final_frame = self.add_performance_overlay(processed_frame, detection_result)

                    # Display the frame
                    cv2.imshow('ASL Hand Sign Detection', final_frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        logger.info("Quit key pressed")
                        break
                    elif key == ord('s'):
                        # Save current frame
                        timestamp = int(time.time())
                        filename = f"asl_detection_{timestamp}.jpg"
                        cv2.imwrite(filename, final_frame)
                        logger.info(f"Frame saved as: {filename}")
                    elif key == ord('l'):
                        # Toggle landmarks
                        self.draw_landmarks = not self.draw_landmarks
                        logger.info(f"Landmarks drawing: {'ON' if self.draw_landmarks else 'OFF'}")
                    elif key == ord('p'):
                        # Print performance stats
                        if self.asl_detector:
                            stats = self.asl_detector.get_performance_stats()
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

        # Cleanup detector
        if self.asl_detector:
            self.asl_detector.cleanup()

        # Components cleanup is handled by context managers
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
            "average_inference_time": self.total_inference_time / max(self.frame_count, 1),
            "detected_signs": len(set(self.detected_signs)),  # Unique signs
            "total_detections": len(self.detected_signs)
        }

        if self.asl_detector:
            detector_stats = self.asl_detector.get_performance_stats()
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
    app = ASLHandSignApp(
        camera_id=0,
        width=640,
        height=480,
        fps=30,
        draw_landmarks=True
    )

    try:
        app.run()

        # Print final performance summary
        summary = app.get_performance_summary()
        logger.info("=" * 55)
        logger.info("Performance Summary:")
        logger.info("=" * 55)
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
