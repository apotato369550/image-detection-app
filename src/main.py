#!/usr/bin/env python3
"""
Computer Vision Application Main Entry Point

This module serves as the main entry point for the computer vision application.
It handles initialization, camera setup, and coordinates the model inference pipeline.
"""

import cv2
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComputerVisionApp:
    """Main application class for computer vision tasks."""

    def __init__(self):
        """Initialize the computer vision application."""
        self.cap = None
        self.running = False

    def initialize_camera(self, camera_id: int = 0) -> bool:
        """
        Initialize camera capture.

        Args:
            camera_id: Camera device ID (default: 0)

        Returns:
            bool: True if camera initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}")
                return False

            logger.info(f"Camera {camera_id} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            return False

    def run(self) -> None:
        """Main application loop."""
        logger.info("Starting Computer Vision Application...")

        if not self.initialize_camera():
            logger.error("Failed to initialize camera. Exiting.")
            sys.exit(1)

        self.running = True

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to grab frame")
                    break

                # TODO: Add model inference here
                # processed_frame = self.process_frame(frame)

                # Display the frame
                cv2.imshow('Computer Vision App', frame)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.running = False

        if self.cap:
            self.cap.release()
            logger.info("Camera released")

        cv2.destroyAllWindows()
        logger.info("Application closed")


def main() -> None:
    """Main entry point function."""
    app = ComputerVisionApp()
    app.run()


if __name__ == "__main__":
    main()