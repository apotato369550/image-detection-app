"""
Configuration Module

Centralized configuration for the ASL Hand Sign Detection Application.
Contains all settings for models, camera, detection, display, and file paths.
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Centralized application configuration."""

    def __init__(self):
        """Initialize configuration with default values."""

        # Project paths
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"

        # Model configuration - Using direct SavedModel download
        self.models = {
            "ssd_mobilenet_v2": {
                "url": "https://storage.googleapis.com/tfhub-modules/tensorflow/ssd_mobilenet_v2/2.tar.gz",
                "filename": "ssd_mobilenet_v2.tar.gz",
                "extracted_name": "",  # Extract to root of models directory
                "model_dir": "",  # Model files are directly in models/
                "model_file": "saved_model.pb",
                "description": "SSD MobileNet V2 - Fast, good for real-time detection",
                "input_size": (320, 320)
            }
        }

        # Default model selection
        self.default_model = "ssd_mobilenet_v2"

        # Camera configuration
        self.camera = {
            "camera_id": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "auto_focus": True,
            "brightness": 128,
            "contrast": 128
        }

        # ASL hand sign detection configuration
        self.detection = {
            "confidence_threshold": 0.7,
            "iou_threshold": 0.45,
            "max_detections": 100,
            "enable_nms": True,
            "nms_method": "default",  # "default", "soft", "hard"
            "hand_detection_only": True,  # Only detect hands, filter out other objects
            "enable_skin_detection": True,  # Use skin color filtering for better hand detection
            "min_hand_size": 1000  # Minimum hand region size in pixels
        }

        # Visualization configuration
        self.visualization = {
            "box_thickness": 2,
            "text_scale": 0.6,
            "text_thickness": 2,
            "font": 0,  # OpenCV font type
            "show_confidence": True,
            "show_class_name": True,
            "show_fps": True,
            "show_inference_time": True,
            "color_palette": "golden_angle"  # "golden_angle", "rainbow", "pastel"
        }

        # Application configuration
        self.app = {
            "window_title": "ASL Hand Sign Detection App",
            "enable_logging": True,
            "log_level": "INFO",
            "log_file": None,  # None for console only, or path for file logging
            "save_frames": True,
            "frame_save_dir": self.project_root / "output",
            "auto_create_dirs": True
        }

        # Dataset configuration
        self.datasets = {
            "asl_alphabet_train": {
                "url": "https://www.kaggle.com/api/v1/datasets/download/grassknoted/asl-alphabet/asl_alphabet_train.csv",
                "filename": "asl_alphabet_train.csv",
                "description": "ASL Alphabet Training Data (A-Z hand signs)",
                "expected_files": [
                    "asl_alphabet_train.csv"
                ]
            },
            "asl_alphabet_test": {
                "url": "https://www.kaggle.com/api/v1/datasets/download/grassknoted/asl-alphabet/asl_alphabet_test.csv",
                "filename": "asl_alphabet_test.csv",
                "description": "ASL Alphabet Test Data (A-Z hand signs)",
                "expected_files": [
                    "asl_alphabet_test.csv"
                ]
            }
        }

        # Performance configuration
        self.performance = {
            "enable_profiling": False,
            "profile_output": "performance.prof",
            "memory_monitoring": False,
            "target_fps": 30,
            "max_frame_skips": 5
        }

        # Create necessary directories
        if self.app["auto_create_dirs"]:
            self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories for the application."""
        directories = [
            self.models_dir,
            self.data_dir,
            self.logs_dir,
            self.app["frame_save_dir"]
        ]

        for directory in directories:
            directory.mkdir(exist_ok=True)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dict containing model configuration
        """
        return self.models.get(model_name, self.models[self.default_model])

    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration."""
        return self.camera.copy()

    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration."""
        return self.detection.copy()

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.visualization.copy()

    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self.app.copy()

    def update_model_config(self, model_name: str, **kwargs) -> None:
        """
        Update configuration for a specific model.

        Args:
            model_name: Name of the model to update
            **kwargs: Configuration parameters to update
        """
        if model_name in self.models:
            self.models[model_name].update(kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def update_camera_config(self, **kwargs) -> None:
        """Update camera configuration."""
        self.camera.update(kwargs)

    def update_detection_config(self, **kwargs) -> None:
        """Update detection configuration."""
        self.detection.update(kwargs)

    def update_visualization_config(self, **kwargs) -> None:
        """Update visualization configuration."""
        self.visualization.update(kwargs)

    def save_config(self, filepath: str = "config.json") -> None:
        """
        Save current configuration to JSON file.

        Args:
            filepath: Path to save configuration file
        """
        import json

        # Convert Path objects to strings for JSON serialization
        config_dict = self._to_dict()

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def load_config(self, filepath: str = "config.json") -> None:
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to configuration file
        """
        import json

        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        self._from_dict(config_dict)

    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        def path_to_str(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: path_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [path_to_str(item) for item in obj]
            else:
                return obj

        return {
            "models": path_to_str(self.models),
            "default_model": self.default_model,
            "camera": self.camera,
            "detection": self.detection,
            "visualization": self.visualization,
            "app": path_to_str(self.app),
            "datasets": self.datasets,
            "performance": self.performance
        }

    def _from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        def str_to_path(obj):
            if isinstance(obj, str) and ('models' in obj or 'data' in obj or 'logs' in obj or 'output' in obj):
                return Path(obj)
            elif isinstance(obj, dict):
                return {k: str_to_path(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [str_to_path(item) for item in obj]
            else:
                return obj

        if "models" in config_dict:
            self.models = str_to_path(config_dict["models"])
        if "default_model" in config_dict:
            self.default_model = config_dict["default_model"]
        if "camera" in config_dict:
            self.camera = config_dict["camera"]
        if "detection" in config_dict:
            self.detection = config_dict["detection"]
        if "visualization" in config_dict:
            self.visualization = config_dict["visualization"]
        if "app" in config_dict:
            self.app = str_to_path(config_dict["app"])
        if "datasets" in config_dict:
            self.datasets = config_dict["datasets"]
        if "performance" in config_dict:
            self.performance = config_dict["performance"]

    def print_config(self) -> None:
        """Print current configuration to console."""
        print("ASL Hand Sign Detection Application Configuration")
        print("=" * 55)

        print("\nModel Configuration:")
        print(f"  Default Model: {self.default_model}")
        for name, config in self.models.items():
            print(f"  {name}: {config['description']}")

        print("\nCamera Configuration:")
        for key, value in self.camera.items():
            print(f"  {key}: {value}")

        print("\nDetection Configuration:")
        for key, value in self.detection.items():
            print(f"  {key}: {value}")

        print("\nVisualization Configuration:")
        for key, value in self.visualization.items():
            print(f"  {key}: {value}")

        print("\nApplication Configuration:")
        for key, value in self.app.items():
            print(f"  {key}: {value}")


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def create_default_config() -> Config:
    """Create a new configuration instance with default values."""
    return Config()


if __name__ == "__main__":
    # Example usage
    config = Config()
    config.print_config()

    # Save configuration
    config.save_config("config.json")
    print("\nConfiguration saved to config.json")