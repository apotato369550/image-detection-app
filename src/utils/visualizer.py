"""
Visualization Module

Provides functions for drawing object detection results on frames.
Features efficient bounding box drawing, class labels, and confidence scores
with optimized performance for real-time applications.
"""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DetectionVisualizer:
    """
    Handles visualization of object detection results.

    Provides efficient methods for drawing bounding boxes, labels, and
    confidence scores with class-specific colors and optimized performance.
    """

    def __init__(self,
                 box_thickness: int = 2,
                 text_scale: float = 0.5,
                 text_thickness: int = 1,
                 font: int = 0) -> None:
        """
        Initialize the DetectionVisualizer.

        Args:
            box_thickness: Thickness of bounding box lines
            text_scale: Scale factor for text size
            text_thickness: Thickness of text lines
            font: OpenCV font type (0-7)
        """
        self.box_thickness = box_thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.font = font

        # Generate color palette for different classes
        self.colors = self._generate_color_palette()

    def _generate_color_palette(self, num_colors: int = 80) -> List[Tuple[int, int, int]]:
        """
        Generate a diverse color palette for different object classes.

        Args:
            num_colors: Number of colors to generate

        Returns:
            List[Tuple[int, int, int]]: RGB color palette
        """
        # Use a deterministic approach to generate visually distinct colors
        colors = []

        for i in range(num_colors):
            # Use golden angle approximation for better color distribution
            hue = (i * 137.508) % 360  # Golden angle in degrees

            # Convert HSV to RGB
            # Saturation and Value are set to maximum for vivid colors
            c = 1.0  # Chroma
            x = c * (1 - abs((hue / 60) % 2 - 1))

            if 0 <= hue < 60:
                r, g, b = c, x, 0
            elif 60 <= hue < 120:
                r, g, b = x, c, 0
            elif 120 <= hue < 180:
                r, g, b = 0, c, x
            elif 180 <= hue < 240:
                r, g, b = 0, x, c
            elif 240 <= hue < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            # Scale to 0-255 range
            color = (int(r * 255), int(g * 255), int(b * 255))
            colors.append(color)

        return colors

    def get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """
        Get color for a specific class ID.

        Args:
            class_id: Class identifier

        Returns:
            Tuple[int, int, int]: RGB color for the class
        """
        # Use modulo to handle any number of classes
        color_index = class_id % len(self.colors)
        return self.colors[color_index]

    def draw_single_detection(self,
                            frame: np.ndarray,
                            detection,
                            draw_bbox: bool = True,
                            draw_label: bool = True) -> np.ndarray:
        """
        Draw a single detection on a frame.

        Args:
            frame: Input frame
            detection: Detection object with bbox, class_name, confidence
            draw_bbox: Whether to draw bounding box
            draw_label: Whether to draw label and confidence

        Returns:
            np.ndarray: Frame with detection drawn
        """
        try:
            import cv2

            # Create a copy to avoid modifying original
            result_frame = frame.copy()

            if draw_bbox:
                # Get box coordinates
                x1, y1, x2, y2 = detection.bbox

                # Get class color
                color = self.get_class_color(detection.class_id)

                # Draw bounding box
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, self.box_thickness)

            if draw_label:
                # Prepare label text
                label = f"{detection.class_name}: {detection.confidence:.2f}"

                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, self.font, self.text_scale, self.text_thickness
                )

                # Position text above the box
                x1, y1, x2, y2 = detection.bbox
                text_x = x1
                text_y = y1 - 10  # 10 pixels above the box

                # Draw background rectangle for text
                cv2.rectangle(
                    result_frame,
                    (text_x, text_y - text_height),
                    (text_x + text_width, text_y + baseline),
                    (0, 0, 0),  # Black background
                    -1  # Filled rectangle
                )

                # Draw text
                cv2.putText(
                    result_frame,
                    label,
                    (text_x, text_y),
                    self.font,
                    self.text_scale,
                    (255, 255, 255),  # White text
                    self.text_thickness,
                    cv2.LINE_AA
                )

            return result_frame

        except ImportError:
            logger.error("OpenCV not installed. Please install with: pip install opencv-python")
            return frame
        except Exception as e:
            logger.error(f"Error drawing detection: {str(e)}")
            return frame

    def draw_detections(self,
                       frame: np.ndarray,
                       detections: List,
                       draw_bbox: bool = True,
                       draw_label: bool = True,
                       max_detections: Optional[int] = None) -> np.ndarray:
        """
        Draw multiple detections on a frame.

        Args:
            frame: Input frame
            detections: List of Detection objects
            draw_bbox: Whether to draw bounding boxes
            draw_label: Whether to draw labels and confidence
            max_detections: Maximum number of detections to draw (None for all)

        Returns:
            np.ndarray: Frame with detections drawn
        """
        try:
            import cv2

            # Create a copy to avoid modifying original
            result_frame = frame.copy()

            # Limit number of detections if specified
            if max_detections is not None:
                detections_to_draw = detections[:max_detections]
            else:
                detections_to_draw = detections

            # Draw each detection
            for detection in detections_to_draw:
                result_frame = self.draw_single_detection(
                    result_frame, detection, draw_bbox, draw_label
                )

            return result_frame

        except ImportError:
            logger.error("OpenCV not installed. Please install with: pip install opencv-python")
            return frame
        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            return frame

    def create_legend(self,
                     frame: np.ndarray,
                     class_names: List[str],
                     max_classes: int = 10) -> np.ndarray:
        """
        Create a legend showing class colors.

        Args:
            frame: Input frame
            class_names: List of class names to include in legend
            max_classes: Maximum number of classes to show

        Returns:
            np.ndarray: Frame with legend added
        """
        try:
            import cv2

            result_frame = frame.copy()

            # Legend parameters
            legend_x = 10
            legend_y = 10
            line_height = 25
            box_size = 15

            # Limit classes to show
            classes_to_show = class_names[:max_classes]

            for i, class_name in enumerate(classes_to_show):
                # Get class color
                class_id = i  # Use index as class_id for color selection
                color = self.get_class_color(class_id)

                # Calculate position
                y_pos = legend_y + i * line_height

                # Draw color box
                cv2.rectangle(
                    result_frame,
                    (legend_x, y_pos),
                    (legend_x + box_size, y_pos + box_size),
                    color,
                    -1  # Filled
                )

                # Draw black border around color box
                cv2.rectangle(
                    result_frame,
                    (legend_x, y_pos),
                    (legend_x + box_size, y_pos + box_size),
                    (0, 0, 0),  # Black border
                    1
                )

                # Draw class name
                cv2.putText(
                    result_frame,
                    class_name,
                    (legend_x + box_size + 5, y_pos + 12),
                    self.font,
                    self.text_scale,
                    (0, 0, 0),  # Black text
                    self.text_thickness,
                    cv2.LINE_AA
                )

            return result_frame

        except ImportError:
            logger.error("OpenCV not installed")
            return frame
        except Exception as e:
            logger.error(f"Error creating legend: {str(e)}")
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


def draw_detection_simple(frame: np.ndarray,
                         x1: int, y1: int, x2: int, y2: int,
                         class_name: str, confidence: float,
                         color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Simple function to draw a single detection.

    Args:
        frame: Input frame
        x1, y1, x2, y2: Bounding box coordinates
        class_name: Name of detected class
        confidence: Confidence score
        color: RGB color for the box

    Returns:
        np.ndarray: Frame with detection drawn
    """
    try:
        import cv2

        result_frame = frame.copy()

        # Draw bounding box
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(result_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                   cv2.LINE_AA)

        return result_frame

    except ImportError:
        logger.error("OpenCV not installed")
        return frame
    except Exception as e:
        logger.error(f"Error in simple drawing: {str(e)}")
        return frame


def draw_detections_batch(frame: np.ndarray,
                         detections: List[Dict]) -> np.ndarray:
    """
    Draw multiple detections from a simple dictionary format.

    Args:
        frame: Input frame
        detections: List of detection dictionaries with keys:
                   'bbox': [x1, y1, x2, y2], 'class_name', 'confidence'

    Returns:
        np.ndarray: Frame with detections drawn
    """
    try:
        import cv2

        result_frame = frame.copy()

        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']

            # Use class name to determine color (simple hash)
            class_id = hash(class_name) % 80
            color = [(class_id * 137) % 255, (class_id * 157) % 255, (class_id * 173) % 255]
            color = (int(color[0]), int(color[1]), int(color[2]))

            # Draw detection
            result_frame = draw_detection_simple(
                result_frame, bbox[0], bbox[1], bbox[2], bbox[3],
                class_name, confidence, color
            )

        return result_frame

    except ImportError:
        logger.error("OpenCV not installed")
        return frame
    except Exception as e:
        logger.error(f"Error in batch drawing: {str(e)}")
        return frame


def main() -> None:
    """Example usage of DetectionVisualizer."""
    # Create a dummy frame for testing
    try:
        import cv2
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create visualizer
        visualizer = DetectionVisualizer()

        print("DetectionVisualizer created successfully")
        print(f"Color palette size: {len(visualizer.colors)}")
        print(f"Box thickness: {visualizer.box_thickness}")
        print(f"Text scale: {visualizer.text_scale}")

        # Test color generation
        for i in range(5):
            color = visualizer.get_class_color(i)
            print(f"Class {i} color: {color}")

    except ImportError:
        print("OpenCV not available for testing")
    except Exception as e:
        print(f"Error in test: {e}")


if __name__ == "__main__":
    main()