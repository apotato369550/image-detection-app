# ASL Hand Sign Detection Application

## Vision & Goals

Build a production-ready, real-time American Sign Language (ASL) alphabet detection system that:
- Accurately recognizes 24 static ASL letters (A-Y, excluding motion-based J and Z)
- Provides immediate visual feedback with hand landmark visualization
- Achieves 20-30 FPS on standard hardware for responsive user experience
- Maintains modular, extensible architecture for future ML classifier integration
- Serves as a foundation for expanded sign language recognition capabilities

**Success Criteria:**
- 80-87% accuracy on well-defined hand poses
- <50ms inference time per frame
- Robust multi-hand support
- Clean separation of concerns for maintainability

## Core Context

### What This Application Does
This is a computer vision application that uses MediaPipe Hands for real-time hand landmark detection (21-point pose estimation) and geometric feature analysis to classify ASL alphabet hand signs from webcam input.

### Technology Stack
- **Python 3.8+**: Core language
- **MediaPipe Hands**: Hand detection and 21-landmark pose estimation
- **OpenCV (cv2)**: Camera handling, frame processing, visualization
- **TensorFlow**: ML framework (for future classifier training)
- **NumPy**: Numerical computations for geometric features

### Current Implementation
- **Detection Method**: Rule-based geometric classifier analyzing distances/angles between hand landmarks
- **Supported Signs**: 24 static ASL letters (A-Y excluding J, Z which require motion)
- **Performance**: ~30-50ms inference, 20-30 FPS on modern hardware
- **Key Entry Points**:
  - `src/asl_main.py` - ASL detection application (primary)
  - `src/main.py` - Legacy COCO object detection (retained for reference)

### Project Structure
```
/
├── CLAUDE.md                    # This file - project context
├── config.py                    # Centralized configuration
├── requirements.txt             # Python dependencies
├── src/
│   ├── asl_main.py             # ASL detection entry point
│   ├── models/
│   │   ├── asl_detector.py     # MediaPipe Hands + ASL classifier
│   │   └── model_manager.py    # Model downloading/caching
│   └── utils/
│       ├── camera.py           # Webcam handling
│       ├── asl_visualizer.py   # Hand landmark + detection visualization
│       └── dataset_manager.py  # Dataset utilities
├── .claude/
│   ├── agents/                 # Agent definitions
│   └── logs/                   # Agent execution logs
└── venv/                       # Virtual environment (local only)
```

## Architectural Principles

### 1. Modular Design
- **Separation of Concerns**: Detection, visualization, camera handling are isolated modules
- **Single Responsibility**: Each class has one clear purpose
- **Dependency Injection**: Components receive dependencies explicitly (no hidden global state)

### 2. Configuration Management
- **Centralized Config**: All settings in `config.py` - never hardcode values
- **Layered Defaults**: Sensible defaults with easy override capability
- **Type Safety**: Use type hints consistently across codebase

### 3. MediaPipe Integration Patterns
- **Lazy Loading**: Initialize MediaPipe models only when needed
- **Resource Cleanup**: Always use context managers or explicit cleanup for MediaPipe resources
- **BGR Format**: MediaPipe expects BGR (OpenCV native), convert to RGB only when needed
- **Performance**: Process at native camera resolution, resize only for display if needed

### 4. Error Handling & Robustness
- **Graceful Degradation**: Camera failures should not crash application
- **Informative Errors**: Log errors with context (frame number, timestamp, config state)
- **Defensive Programming**: Validate inputs, check for None/empty results
- **Resource Limits**: Set reasonable maximums for detections, history buffers, etc.

### 5. Performance Optimization
- **Minimize Copies**: Reuse frame buffers, avoid unnecessary image copies
- **Vectorized Operations**: Use NumPy for batch calculations
- **Profiling-Driven**: Only optimize based on measured bottlenecks
- **Target Metrics**: Maintain 20+ FPS, <50ms inference time

### 6. Code Quality Standards
- **Documentation**: Docstrings for all public functions with type hints
- **Naming Conventions**: Clear, descriptive names following PEP 8
- **Import Organization**: Standard library, third-party, local (groups separated)
- **Keep It Simple**: Prefer readability over cleverness

## Future Improvement Paths

### Phase 1: ML Classifier Integration (Next Priority)
- Replace rule-based classifier with trained ML model (Random Forest, SVM, or small NN)
- Use MediaPipe landmarks as feature vectors for training
- Leverage ASL Alphabet datasets from Kaggle for training data
- Expected accuracy improvement: 80-87% → 95%+

### Phase 2: Temporal Analysis
- Track hand poses across frames for consistency filtering
- Implement motion detection for J and Z signs
- Add confidence smoothing using rolling averages
- Detect sign transitions and hold times

### Phase 3: Expanded Sign Library
- Add ASL numbers (0-9)
- Implement common words and phrases
- Support two-handed signs
- Build sign vocabulary database

### Phase 4: Real-Time Translation
- Convert detected signs to text stream
- Implement word completion and autocorrection
- Add text-to-speech output
- Build conversational interface

## Critical Dependencies & Constraints

### Hardware Requirements
- **Webcam**: Minimum 640x480 resolution, 15+ FPS
- **CPU**: Modern multi-core processor (MediaPipe is CPU-optimized)
- **RAM**: Minimum 4GB (MediaPipe models + OpenCV buffers)
- **GPU**: Optional, not required for MediaPipe Hands

### Software Dependencies
- **Python 3.8+**: Minimum version for MediaPipe compatibility
- **OpenCV**: Camera interface and visualization
- **MediaPipe**: Hand detection (auto-downloads models on first run)
- **TensorFlow**: Required by MediaPipe, future ML classifier
- **NumPy**: Geometric calculations

### Known Limitations
- **Motion Signs**: Cannot detect J and Z (require temporal analysis)
- **Lighting Sensitivity**: Performance degrades in poor lighting
- **Occlusion**: Partial hand visibility reduces accuracy
- **Hand Orientation**: Works best with palm-toward-camera angles
- **Classifier Method**: Rule-based system has accuracy ceiling (~87%)

## Agent Coordination Guidelines

### Multi-Agent Development Pattern
This project uses a coordinated multi-agent system. Agents must:
- **Read CLAUDE.md first** to understand project context and principles
- **Respect architectural patterns** (modular design, centralized config)
- **Log work concisely** in designated agent directories
- **Coordinate through Foreman** for complex multi-component changes

### When Making Changes
1. **Check Dependencies**: Will this affect MediaPipe, OpenCV, or TensorFlow versions?
2. **Test Impact**: Changes to core detector affect entire application
3. **Document Config**: New settings go in `config.py` with descriptions
4. **Update Docs**: User-facing changes require README updates

### Debugging Protocol
- **Environment First**: Validate venv, dependencies, Python version
- **Isolate Components**: Test camera, MediaPipe, classifier separately
- **Check Logs**: Enable debug logging in config for detailed traces
- **Profile Performance**: Use FPS metrics to identify bottlenecks

### Quality Gates
- **Before Implementation**: Plan must account for MediaPipe resource management
- **Before Validation**: Code must follow PEP 8 and include docstrings
- **Before Deployment**: Test with real webcam under various lighting conditions
- **Before PR**: Ensure no breaking changes to existing ASL detection

## Technical Debt & Known Issues

### Current Technical Debt
1. **Rule-Based Classifier**: Limited by geometric heuristics, needs ML replacement
2. **No Unit Tests**: Testing is manual via webcam (needs automated test suite)
3. **No CI/CD**: Dependency management is manual (need automated testing)
4. **Limited Error Recovery**: Camera failures should retry with backoff
5. **Configuration Complexity**: Some config options unused/undocumented

### Planned Refactoring
- Extract feature calculation from ASLHandDetector into separate FeatureExtractor class
- Implement proper logging framework (currently print statements)
- Add configuration validation on startup
- Create reproducible test datasets (recorded frames with ground truth)

## Development Environment

### Virtual Environment Setup
```bash
# Create venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# ASL detection (primary application)
python src/asl_main.py

# Legacy COCO detection (for reference)
python src/main.py
```

### Key Configuration Points
- `config.py`: Main configuration file
- MediaPipe downloads models to `~/.mediapipe/` on first run
- Output frames saved to `output/` directory
- Camera settings: 640x480 @ 30 FPS (adjust in config for performance)

---

**This document is the authoritative source of truth for project context. All agents should reference this before beginning work.**
