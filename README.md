# Object Detection Application

A complete, production-ready Python computer vision application for real-time object detection using TensorFlow and OpenCV. Features automatic model downloading, webcam integration, and live visualization of detected objects.

## ⚠️ Important Note About ASL Hand Sign Detection

**This application currently uses a pre-trained COCO dataset model (SSD MobileNet V2), which detects general objects like people, cars, animals, etc., but does NOT detect ASL hand signs.**

To detect ASL hand signs, you would need:
1. A model specifically trained on ASL hand sign data
2. A custom-trained TensorFlow/PyTorch model with ASL alphabet classes
3. Dataset like ASL-MNIST, ASL Alphabet Dataset, or similar

**The current model can detect 90 COCO object classes including:**
- person, bicycle, car, motorcycle, airplane, bus, train, truck, boat
- traffic light, fire hydrant, stop sign, parking meter, bench
- bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- backpack, umbrella, handbag, tie, suitcase, frisbee, sports ball
- bottle, cup, fork, knife, spoon, bowl, banana, apple, sandwich
- chair, couch, bed, dining table, tv, laptop, mouse, keyboard, cell phone
- And many more common objects...

## 🚀 Features

- **Real-time Object Detection**: Live object recognition using pre-trained TensorFlow models
- **Automatic Model Management**: Downloads and caches SSD MobileNet V2 model from TensorFlow Hub
- **Webcam Integration**: Robust camera handling with frame rate control
- **Live Visualization**: Bounding boxes, object labels, and confidence scores
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
│   │   └── detector.py          # Object detection inference (COCO dataset)
│   └── utils/
│       ├── __init__.py          # Utils package
│       ├── camera.py            # Webcam handling
│       ├── dataset_manager.py   # Dataset downloading
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
1. Downloads SSD MobileNet V2 model (~15MB) from TensorFlow Hub
2. Initializes webcam and object detector
3. Starts real-time object detection
4. Displays live video with detected objects (person, bottle, cell phone, etc.)

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

### Basic Object Detection

```python
from src.models.detector import ObjectDetector
from src.utils.camera import CameraHandler
from src.utils.visualizer import DetectionVisualizer
import cv2

# Initialize components
camera = CameraHandler()
detector = ObjectDetector()
visualizer = DetectionVisualizer()

# Load model
detector.load_model("models")

# Process frames
with camera:
    for frame in camera.get_frame_generator():
        # Run object detection (detects COCO objects: person, car, etc.)
        result = detector.detect(frame)

        # Draw detection results
        output_frame = visualizer.draw_detections(frame, result.detections)

        # Display detection results
        cv2.imshow('Object Detection', output_frame)
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

### Supported Object Classes

The SSD MobileNet V2 model can detect 90 COCO dataset classes:

```python
from src.models.detector import ObjectDetector

detector = ObjectDetector()
print("Supported classes:", detector.class_names)
# Output: ['person', 'bicycle', 'car', 'motorcycle', ...]
```

**Most commonly detected objects:**
- People and body parts
- Vehicles: car, bicycle, motorcycle, bus, truck, airplane
- Animals: dog, cat, bird, horse, cow, sheep
- Furniture: chair, couch, bed, dining table
- Electronics: laptop, cell phone, tv, keyboard, mouse
- Kitchen items: bottle, cup, fork, knife, bowl
- Food: banana, apple, sandwich, orange, pizza
- And many more!

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

## 🎯 How to Add ASL Hand Sign Detection

To convert this into a true ASL hand sign detector, you would need to:

1. **Obtain an ASL-trained model:**
   - Train a model on ASL datasets (ASL-MNIST, ASL Alphabet, etc.)
   - Use transfer learning from a pre-trained model
   - Fine-tune on ASL hand sign images

2. **Replace the model:**
   - Update `config.py` with ASL model URL
   - Modify `detector.py` class names to ASL alphabet (A-Z, 0-9, etc.)
   - Adjust preprocessing for hand-specific detection

3. **Recommended ASL datasets:**
   - ASL Alphabet Dataset (Kaggle)
   - ASL-MNIST
   - MS-ASL (Microsoft American Sign Language Dataset)
   - WLASL (World-Level American Sign Language)

## 🙏 Acknowledgments

- **TensorFlow**: Machine learning framework
- **TensorFlow Hub**: Pre-trained SSD MobileNet V2 model
- **OpenCV**: Computer vision library
- **COCO Dataset**: Common Objects in Context dataset for object detection

---

**Ready to detect objects? Run `python src/main.py` and start recognizing everyday objects!**

**Note:** This application currently detects COCO objects (person, car, bottle, etc.), not ASL hand signs. See the section above for how to add ASL detection capability.