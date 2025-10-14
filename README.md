# Computer Vision Application

A modular Python computer vision application built with TensorFlow Lite and OpenCV for efficient real-time image processing and object detection.

## Project Structure

```
image-detection-app/
├── src/
│   ├── __init__.py          # Main package initialization
│   ├── main.py             # Application entry point
│   ├── models/             # Model-related modules
│   │   └── __init__.py
│   └── utils/              # Utility functions
│       └── __init__.py
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Features

- **Modular Architecture**: Clean separation of concerns with models, utils, and main application code
- **TensorFlow Lite Integration**: Efficient inference using TFLite models
- **OpenCV Camera Support**: Real-time camera capture and processing
- **Extensible Design**: Easy to add new models and processing pipelines

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python -c "import cv2, tensorflow as tf; print('Dependencies installed successfully')"
   ```

## Usage

### Basic Usage

Run the main application:
```bash
python src/main.py
```

The application will:
- Initialize the camera (default camera ID: 0)
- Display a live video feed
- Exit when 'q' key is pressed

### Project Organization

- **`src/main.py`**: Main application entry point with camera handling and main loop
- **`src/models/`**: Place your model loading and inference code here
- **`src/utils/`**: Utility functions for image processing, data handling, etc.
- **`requirements.txt`**: All necessary dependencies

## Development

### Adding New Models

1. Create model files in `src/models/`
2. Implement model loading and inference functions
3. Integrate with the main application loop in `src/main.py`

### Adding Utility Functions

1. Create utility modules in `src/utils/`
2. Import and use in other parts of the application

## Dependencies

- **tensorflow-lite**: Efficient model inference
- **opencv-python**: Camera capture and image processing
- **numpy**: Numerical computations
- **pillow**: Image manipulation
- **requests**: HTTP requests for model downloads
- **matplotlib**: Plotting and visualization
- **scipy**: Scientific computing utilities

## Architecture Notes

- Follows Python src layout pattern for better package management
- Designed for standalone applications (not Jupyter notebooks)
- Uses requests library for model downloads (wget functionality)
- Implements SOLID principles for maintainable code
- No testing framework included (manual testing only)