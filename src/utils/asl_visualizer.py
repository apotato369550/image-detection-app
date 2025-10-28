"""
ASL Detection Visualizer Module

Provides functions for drawing ASL hand detection results on frames,
including hand landmarks, bounding boxes, and recognized letters.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ASLVisualizer:
    """
    Handles visualization of ASL hand detection results.

    Provides methods for drawing hand landmarks, bounding boxes, and
    recognized ASL letters with confidence scores.
    """

    def __init__(self,
                 box_thickness: int = 2,
                 text_scale: float = 0.7,
                 text_thickness: int = 2,
                 landmark_size: int = 3):
        """
        Initialize the ASL Visualizer.

        Args:
            box_thickness: Thickness of bounding box lines
            text_scale: Scale factor for text size
            text_thickness: Thickness of text lines
            landmark_size: Size of landmark circles
        """
        self.box_thickness = box_thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.landmark_size = landmark_size

        # Color scheme for visualization
        self.color_landmark = (0, 255, 0)  # Green
        self.color_connection = (200, 200, 0)  # Cyan
        self.color_bbox_left = (255, 0, 0)  # Blue (left hand)
        self.color_bbox_right = (0, 165, 255)  # Orange (right hand)
        self.color_text_bg = (0, 0, 0)  # Black
        self.color_text = (255, 255, 255)  # White

    def draw_hand_landmarks(self,
                           frame: np.ndarray,
                           landmarks_2d: List[Tuple[float, float]],
                           handedness: str = "Unknown") -> np.ndarray:
        """
        Draw hand landmarks on frame.

        Args:
            frame: Input frame
            landmarks_2d: List of (x, y) normalized landmark coordinates
            handedness: "Left" or "Right" hand

        Returns:
            Frame with landmarks drawn
        """
        try:
            result_frame = frame.copy()
            h, w = frame.shape[:2]

            # Hand landmark connection indices (MediaPipe format)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
            ]

            # Convert normalized coordinates to pixel coordinates
            pixel_landmarks = [
                (int(lm[0] * w), int(lm[1] * h))
                for lm in landmarks_2d
            ]

            # Draw connections
            for start, end in connections:
                if start < len(pixel_landmarks) and end < len(pixel_landmarks):
                    pt1 = pixel_landmarks[start]
                    pt2 = pixel_landmarks[end]
                    cv2.line(result_frame, pt1, pt2, self.color_connection, 1)

            # Draw landmarks
            for i, pt in enumerate(pixel_landmarks):
                cv2.circle(result_frame, pt, self.landmark_size, self.color_landmark, -1)

            return result_frame

        except Exception as e:
            logger.error(f"Error drawing landmarks: {str(e)}")
            return frame

    def draw_asl_detection(self,
                          frame: np.ndarray,
                          letter: str,
                          confidence: float,
                          bbox: Tuple[int, int, int, int],
                          handedness: str = "Unknown",
                          landmarks_2d: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """
        Draw a single ASL detection on frame.

        Args:
            frame: Input frame
            letter: Detected ASL letter
            confidence: Confidence score
            bbox: Bounding box (x1, y1, x2, y2)
            handedness: "Left" or "Right" hand
            landmarks_2d: Hand landmarks (optional)

        Returns:
            Frame with detection drawn
        """
        try:
            result_frame = frame.copy()
            x1, y1, x2, y2 = bbox

            # Choose color based on handedness
            if handedness == "Left":
                color = self.color_bbox_left
            else:
                color = self.color_bbox_right

            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, self.box_thickness)

            # Draw hand landmarks if provided
            if landmarks_2d is not None:
                result_frame = self.draw_hand_landmarks(result_frame, landmarks_2d, handedness)

            # Prepare label text
            label = f"{letter}: {confidence:.2f} ({handedness})"

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
            )

            # Position text above the box
            text_x = x1
            text_y = y1 - 10

            # Draw background rectangle for text
            cv2.rectangle(
                result_frame,
                (text_x, text_y - text_height - 5),
                (text_x + text_width + 5, text_y + baseline),
                self.color_text_bg,
                -1
            )

            # Draw text
            cv2.putText(
                result_frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                self.color_text,
                self.text_thickness,
                cv2.LINE_AA
            )

            return result_frame

        except Exception as e:
            logger.error(f"Error drawing detection: {str(e)}")
            return frame

    def draw_asl_detections(self,
                           frame: np.ndarray,
                           detections: List,
                           draw_landmarks: bool = True) -> np.ndarray:
        """
        Draw multiple ASL detections on frame.

        Args:
            frame: Input frame
            detections: List of ASL detection objects
            draw_landmarks: Whether to draw hand landmarks

        Returns:
            Frame with detections drawn
        """
        try:
            result_frame = frame.copy()

            for detection in detections:
                landmarks_2d = detection.hand_landmarks if draw_landmarks else None
                result_frame = self.draw_asl_detection(
                    result_frame,
                    detection.letter,
                    detection.confidence,
                    detection.bbox,
                    detection.handedness,
                    landmarks_2d
                )

            return result_frame

        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            return frame

    def draw_asl_legend(self,
                       frame: np.ndarray,
                       detected_letters: List[str]) -> np.ndarray:
        """
        Draw a legend showing detected ASL letters on frame.

        Args:
            frame: Input frame
            detected_letters: List of detected letters

        Returns:
            Frame with legend added
        """
        try:
            result_frame = frame.copy()

            # Legend parameters
            legend_x = 10
            legend_y = 60
            line_height = 30

            # Title
            cv2.putText(
                result_frame,
                "Detected Signs:",
                (legend_x, legend_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.color_text,
                2,
                cv2.LINE_AA
            )

            # Draw detected letters
            for i, letter in enumerate(detected_letters[:6]):  # Limit to 6 letters
                y_pos = legend_y + i * line_height

                cv2.putText(
                    result_frame,
                    f"  {letter}",
                    (legend_x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    self.color_text,
                    2,
                    cv2.LINE_AA
                )

            return result_frame

        except Exception as e:
            logger.error(f"Error drawing legend: {str(e)}")
            return frame

    def draw_info_overlay(self,
                         frame: np.ndarray,
                         fps: float,
                         num_hands: int) -> np.ndarray:
        """
        Draw performance and status information overlay.

        Args:
            frame: Input frame
            fps: Current FPS
            num_hands: Number of hands detected

        Returns:
            Frame with info overlay
        """
        try:
            result_frame = frame.copy()

            # FPS info
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                result_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_text,
                2,
                cv2.LINE_AA
            )

            # Hands detected info
            hands_text = f"Hands: {num_hands}"
            cv2.putText(
                result_frame,
                hands_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_text,
                2,
                cv2.LINE_AA
            )

            return result_frame

        except Exception as e:
            logger.error(f"Error drawing info: {str(e)}")
            return frame

    def set_box_thickness(self, thickness: int) -> None:
        """Set bounding box thickness."""
        self.box_thickness = max(1, thickness)

    def set_text_scale(self, scale: float) -> None:
        """Set text scale factor."""
        self.text_scale = max(0.1, scale)

    def set_text_thickness(self, thickness: int) -> None:
        """Set text thickness."""
        self.text_thickness = max(1, thickness)


def main() -> None:
    """Example usage of ASLVisualizer."""
    visualizer = ASLVisualizer()
    print("ASL Visualizer created successfully")
    print(f"Box thickness: {visualizer.box_thickness}")
    print(f"Text scale: {visualizer.text_scale}")


if __name__ == "__main__":
    main()
