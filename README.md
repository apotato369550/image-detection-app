# ASL Hand Sign Detection Application

A complete, production-ready Python computer vision application for real-time American Sign Language (ASL) hand sign detection using TensorFlow Lite and OpenCV. Features automatic model downloading, webcam integration, and live visualization of ASL alphabet and common signs.

## 🚀 Features

- **Real-time ASL Hand Sign Detection**: Live American Sign Language hand sign recognition using pretrained TensorFlow Lite models
- **Automatic Model Management**: Downloads and caches models suitable for hand sign detection
- **Webcam Integration**: Robust camera handling with frame rate control
- **Live Visualization**: Bounding boxes, ASL sign labels, and confidence scores
- **Performance Monitoring**: Real-time FPS and inference speed tracking
- **Modular Architecture**: Clean separation of concerns with extensible design
- **Configuration Management**: Centralized configuration for easy customization

## 📁 Project Structure

```
image-detection-app/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Main application entry point
│   ├── models/
│   │   ├── __init__.py          # Models package
│   │   ├── model_manager.py     # Model downloading and caching
│   │   └── detector.py          # ASL hand sign detection inference
│   └── utils/
│       ├── __init__.py          # Utils package
│       ├── camera.py            # Webcam handling
│       ├── dataset_manager.py   # ASL dataset downloading
│       └── visualizer.py        # Detection visualization
├── config.py                       # Centralized configuration
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package metadata (optional)
└── README.md                        # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for live detection)
- Internet connection (for model downloads)

### Step 1: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import cv2, tensorflow as tf; print('✅ Dependencies installed successfully')"
```

## 🚀 Quick Start

### Run the Complete Application

```bash
# From project root directory
python src/main.py

# Or using the installed package (if installed with pip install -e .)
cv-app
```

**What happens on first run:**
1. Downloads SSD MobileNet V2 model (~15MB) suitable for hand sign detection
2. Initializes webcam and ASL hand sign detector
3. Starts real-time ASL hand sign detection
4. Displays live video with ASL sign recognition results

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save current frame as image |
| `p` | Print performance statistics |

## ⚙️ Configuration

The application uses a centralized configuration system. You can customize:

### Basic Configuration

```python
from config import config

# Change model
config.default_model = "efficientdet_d0"

# Adjust detection threshold
config.detection["confidence_threshold"] = 0.3

# Change camera resolution
config.camera["width"] = 1280
config.camera["height"] = 720

# Save configuration
config.save_config("my_config.json")
```

### Configuration Options

#### Model Settings
- **Model Selection**: Choose from available pretrained models
- **Confidence Threshold**: Minimum detection confidence (0.0-1.0)
- **IoU Threshold**: Non-maximum suppression threshold

#### Camera Settings
- **Resolution**: Frame width and height
- **Frame Rate**: Target FPS for capture
- **Camera Index**: Select different cameras (0, 1, 2, etc.)

#### Visualization Settings
- **Box Thickness**: Bounding box line thickness
- **Text Scale**: Size of labels and confidence scores
- **Color Palette**: Color scheme for different classes

## 📖 Usage Examples

### Basic ASL Hand Sign Detection

```python
from src.models.detector import ObjectDetector
from src.utils.camera import CameraHandler
from src.utils.visualizer import DetectionVisualizer

# Initialize components
camera = CameraHandler()
detector = ObjectDetector()
visualizer = DetectionVisualizer()

# Process frames
with camera:
    for frame in camera.get_frame_generator():
        # Run ASL hand sign detection
        result = detector.detect(frame)

        # Draw ASL sign results
        output_frame = visualizer.draw_detections(frame, result.detections)

        # Display ASL detection results
        cv2.imshow('ASL Hand Sign Detection', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

### Model Management

```python
from src.models.model_manager import ModelManager

# Initialize manager
manager = ModelManager()

# List available models
models = manager.list_available_models()
print("Available models:", models)

# Download specific model
model_path = manager.download_model("ssd_mobilenet_v2")
print(f"Model downloaded to: {model_path}")
```

### ASL Dataset Handling

```python
from src.utils.dataset_manager import DatasetManager

# Initialize manager
manager = DatasetManager()

# Download ASL alphabet dataset
dataset_path = manager.download_dataset("asl_alphabet_test")

# Get ASL class names
class_names = manager.get_class_names("asl_alphabet_test.json")
print(f"ASL signs: {class_names[:10]}...")
```

## 🔧 Advanced Configuration

### Custom Model Integration

To add a new model:

1. **Add to config.py**:
```python
config.models["my_model"] = {
    "url": "https://example.com/model.tar.gz",
    "filename": "model.tar.gz",
    "model_dir": "model/saved_model",
    "description": "My custom model",
    "input_size": (320, 320)
}
```

2. **Update model manager** if needed for different download format

### Performance Optimization

```python
# Adjust for your hardware
config.detection["max_detections"] = 50  # Reduce for slower hardware
config.camera["fps"] = 15               # Lower FPS for slower inference
config.models["ssd_mobilenet_v2"]["input_size"] = (224, 224)  # Smaller input
```

## 📊 Performance

### Expected Performance

| Model | Input Size | Expected FPS | Accuracy |
|-------|------------|--------------|----------|
| SSD MobileNet V2 | 320x320 | 20-30 FPS | Good |
| EfficientDet D0 | 512x512 | 10-15 FPS | Better |

*Performance varies based on hardware. Tested on mid-range GPU.*

### Performance Monitoring

The application displays real-time performance metrics:
- **FPS**: Frame processing rate
- **Inference FPS**: Model inference speed
- **Frame Count**: Total processed frames

## 🐛 Troubleshooting

### Common Issues

#### 1. Camera Not Found
```
Error: Failed to open camera 0
```
**Solutions:**
- Check if camera is connected
- Try different camera index: `config.camera["camera_id"] = 1`
- List available cameras: `python -c "from src.utils.camera import list_available_cameras; print(list_available_cameras())"`

#### 2. Model Download Fails
```
Error: Download failed
```
**Solutions:**
- Check internet connection
- Verify model URL in configuration
- Check available disk space
- Try downloading manually and place in `models/` directory

#### 3. Low Performance
```
FPS: 5.2 | Inference: 3.1 FPS
```
**Solutions:**
- Reduce input resolution in config
- Lower confidence threshold
- Use SSD MobileNet V2 instead of EfficientDet
- Close other applications

#### 4. Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Solutions:**
- Make sure you're running from the project root directory
- Use relative imports: Run `python src/main.py` from project root
- Check that all `__init__.py` files exist in src directories

```
ModuleNotFoundError: No module named 'tensorflow'
```
**Solutions:**
- Activate virtual environment: `source venv/bin/activate`
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version compatibility

#### 5. Model Loading Issues
```
Error: Could not open 'models\ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model'
```
**Solutions:**
- Delete the models directory and let the application re-download
- Check available disk space for model extraction
- Verify model URL accessibility
- The application will automatically re-download if model is corrupted

#### 6. Display Issues
```
Error: Can't open display window
```
**Solutions:**
- Ensure you're not running in headless environment
- Check OpenCV installation
- Try running with different backend

### Getting Help

1. **Check Logs**: Enable debug logging in config
2. **Verify Installation**: Run the verification command above
3. **Test Components**: Test individual modules separately
4. **Check Hardware**: Ensure webcam and GPU (if used) are working

## 🤝 Contributing

### Project Structure Guidelines

- **src/models/**: Model loading, inference, and management
- **src/utils/**: Utility functions and helpers
- **config.py**: All configuration settings
- **Tests**: Add manual testing examples

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters
- Add docstrings for public functions
- Keep functions focused and concise

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **TensorFlow**: Machine learning framework
- **OpenCV**: Computer vision library
- **ASL Alphabet Dataset**: American Sign Language hand sign dataset
- **TensorFlow Model Zoo**: Pretrained models adapted for ASL detection

---

**Ready to detect ASL hand signs? Run `python src/main.py` and start recognizing American Sign Language!**