"""
Camera Handler Module

Provides CameraHandler class for efficient webcam interaction using OpenCV.
Handles camera initialization, frame capture, and resource management with
proper cleanup and frame rate control.
"""

import logging
import time
from typing import Optional, Tuple, Generator, Union, List, TYPE_CHECKING
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CameraHandler:
    """
    Handles webcam interaction with proper resource management.

    Provides methods for camera initialization, frame capture, and cleanup.
    Supports frame rate control and multiple camera indices.
    """

    def __init__(self,
                 camera_id: int = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30) -> None:
        """
        Initialize the CameraHandler.

        Args:
            camera_id: Camera device index (0 for default camera)
            width: Desired frame width
            height: Desired frame height
            fps: Desired frame rate
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        # Camera state
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.frame_count = 0
        self.start_time = 0.0

        # Frame timing
        self.last_frame_time = 0.0
        self.frame_interval = 1.0 / fps if fps > 0 else 0

    def initialize(self) -> bool:
        """
        Initialize the camera connection.

        Returns:
            bool: True if camera initialized successfully, False otherwise
        """
        try:
            import cv2

            logger.info(f"Initializing camera {self.camera_id}")

            # Create VideoCapture object
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False

            # Set camera properties
            success = True
            success &= self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            success &= self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            success &= self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            if not success:
                logger.warning("Failed to set some camera properties")

            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            logger.info(f"Camera {self.camera_id} initialized successfully")
            logger.info(f"Resolution: {actual_width}x{actual_height}")
            logger.info(f"FPS: {actual_fps}")

            self.is_connected = True
            self.start_time = time.time()
            self.frame_count = 0

            return True

        except ImportError:
            logger.error("OpenCV not installed. Please install with: pip install opencv-python")
            return False
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            return False

    def is_camera_connected(self) -> bool:
        """
        Check if camera is connected and working.

        Returns:
            bool: True if camera is connected, False otherwise
        """
        if self.cap is None:
            return False

        return self.cap.isOpened() and self.is_connected

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the camera.

        Returns:
            Optional[np.ndarray]: Frame as numpy array, None if failed
        """
        if not self.is_camera_connected():
            logger.error("Camera not connected")
            return None

        try:
            import cv2

            # Control frame rate
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                return None

            ret, frame = self.cap.read()
            self.last_frame_time = current_time

            if not ret:
                logger.warning("Failed to grab frame")
                return None

            self.frame_count += 1
            return frame

        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            return None

    def get_frame_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames continuously.

        Yields:
            np.ndarray: Camera frames

        Raises:
            RuntimeError: If camera is not connected
        """
        if not self.is_camera_connected():
            raise RuntimeError("Camera not connected. Call initialize() first.")

        logger.info("Starting frame generator")

        try:
            while self.is_connected:
                frame = self.read_frame()
                if frame is not None:
                    yield frame

        except KeyboardInterrupt:
            logger.info("Frame generator interrupted by user")
        except Exception as e:
            logger.error(f"Error in frame generator: {str(e)}")
        finally:
            logger.info("Frame generator stopped")

    def get_frame_rate(self) -> float:
        """
        Calculate actual frame rate.

        Returns:
            float: Actual FPS achieved
        """
        if self.frame_count == 0 or self.start_time == 0:
            return 0.0

        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0.0

        return self.frame_count / elapsed_time

    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set camera resolution.

        Args:
            width: New width
            height: New height

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_camera_connected():
            logger.error("Camera not connected")
            return False

        try:
            import cv2

            success = True
            success &= self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            success &= self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            if success:
                self.width = width
                self.height = height
                logger.info(f"Resolution set to {width}x{height}")
            else:
                logger.warning("Failed to set resolution")

            return success

        except Exception as e:
            logger.error(f"Error setting resolution: {str(e)}")
            return False

    def set_frame_rate(self, fps: int) -> bool:
        """
        Set target frame rate.

        Args:
            fps: Target FPS

        Returns:
            bool: True if successful, False otherwise
        """
        if fps <= 0:
            logger.error("FPS must be positive")
            return False

        self.fps = fps
        self.frame_interval = 1.0 / fps

        if self.is_camera_connected():
            try:
                import cv2
                success = self.cap.set(cv2.CAP_PROP_FPS, fps)
                if success:
                    logger.info(f"Frame rate set to {fps} FPS")
                else:
                    logger.warning("Failed to set frame rate")
                return success
            except Exception as e:
                logger.error(f"Error setting frame rate: {str(e)}")
                return False

        return True

    def get_camera_info(self) -> dict:
        """
        Get camera information and statistics.

        Returns:
            dict: Camera information including resolution, FPS, frame count
        """
        info = {
            "camera_id": self.camera_id,
            "is_connected": self.is_connected,
            "width": self.width,
            "height": self.height,
            "target_fps": self.fps,
            "actual_fps": self.get_frame_rate(),
            "frame_count": self.frame_count,
            "uptime": time.time() - self.start_time if self.start_time > 0 else 0
        }

        if self.is_camera_connected():
            try:
                import cv2
                info.update({
                    "actual_width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "actual_height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "actual_fps": int(self.cap.get(cv2.CAP_PROP_FPS))
                })
            except Exception:
                pass

        return info

    def disconnect(self) -> None:
        """Disconnect from camera and cleanup resources."""
        logger.info("Disconnecting camera")

        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("Camera released successfully")
            except Exception as e:
                logger.error(f"Error releasing camera: {str(e)}")

        self.cap = None
        self.is_connected = False
        self.frame_count = 0
        self.start_time = 0.0

    def __enter__(self) -> 'CameraHandler':
        """Context manager entry."""
        if not self.initialize():
            raise RuntimeError(f"Failed to initialize camera {self.camera_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()


def list_available_cameras(max_cameras: int = 5) -> List[int]:
    """
    List available camera devices.

    Args:
        max_cameras: Maximum number of cameras to check

    Returns:
        List[int]: List of available camera indices
    """
    available_cameras = []

    try:
        import cv2

        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()

        logger.info(f"Found {len(available_cameras)} available cameras: {available_cameras}")

    except ImportError:
        logger.error("OpenCV not installed")
    except Exception as e:
        logger.error(f"Error checking cameras: {str(e)}")

    return available_cameras


def main() -> None:
    """Example usage of CameraHandler."""
    print("Available cameras:", list_available_cameras())

    # Example usage with context manager
    try:
        with CameraHandler(camera_id=0, width=640, height=480, fps=30) as camera:
            print(f"Camera info: {camera.get_camera_info()}")

            # Get a few frames
            for i, frame in enumerate(camera.get_frame_generator()):
                if i >= 5:  # Just get 5 frames for testing
                    break

                print(f"Frame {i+1} shape: {frame.shape}")
                print(f"Frame rate: {camera.get_frame_rate():.2f} FPS")

    except RuntimeError as e:
        print(f"Camera error: {e}")
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()