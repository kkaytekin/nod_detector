# Nod Detector 👋

A Python package for detecting nodding behavior in videos using MediaPipe. Easily analyze head movements and identify nodding patterns in your video files.

## 🚀 Installation

### Prerequisites
- Python 3.x
- MediaPipe

### Setup
```bash
# Clone the repository
git clone https://github.com/kkaytekin/nod_detector.git
cd nod_detector

# Create and activate a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate  # On Windows
# On Unix/macOS: source venv/bin/activate

# Install the package in development mode with all dependencies
pip install -e .

# For development, install with additional development dependencies
# pip install -e ".[dev]"
```

## 🎬 Quick Start

Process a video file with a single command:
```bash
# Process a video file
nod-detector --input path/to/input_video.mp4 --output output.mp4
```

### Expected Output
- Processed video file with visualizations
- Console output showing processing statistics
- (Future) JSON file with detailed nod detection data

## 📁 Project Structure
```
nod_detector/
├── src/
│   └── nod_detector/     # Main package
│       ├── pipeline/     # Video processing pipeline
│       ├── main.py       # Command-line interface
│       └── __init__.py   # Package definition
├── tests/                # Test suite
└── examples/             # Example scripts
```

## 🔍 Assumptions
- The input video contains clear frontal or near-frontal views of faces
- Lighting conditions are sufficient for face detection
- The subject's head is visible for most of the video duration

## 🤝 Contributing

### For Users
Found a bug or have a feature request? Please open an issue on our [GitHub Issues](https://github.com/kkaytekin/nod_detector/issues) page.

### For Developers
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Set up pre-commit hooks (see below)
4. Make your changes and commit them
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Setup
```bash
# Install the package in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (runs automatically on git commit)
pre-commit install

# Run tests
pytest

# Format code with black
black .

# Check code style with flake8
flake8  # Configured to exclude .venv and other common directories

# Run type checking with mypy
mypy src/
```

### 🔩 Pre-commit Hooks
This project uses pre-commit to run several code quality checks before each commit. The following hooks are configured:

- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting
- **Mypy**: Static type checking
- **Pre-commit hooks**: Various checks for common issues

These hooks run automatically when you make a commit. If any checks fail, the commit will be aborted and you'll need to fix the issues before committing.

### 🚀 Development Workflow
1. Write tests for new features or bug fixes
2. Implement the feature/fix to make the tests pass
3. Run tests and fix any issues (`pytest`)
4. Format your code (`black .`)
5. Check for code style issues (`flake8`)
6. Run type checking (`mypy src/`)
7. Stage and commit your changes - pre-commit hooks will run automatically
8. Push to your fork and open a pull request

### 🧪 Testing

The project includes a comprehensive test suite to ensure reliability and maintainability.

#### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/ -m integration

# Run tests with coverage report
pytest --cov=src/nod_detector --cov-report=term-missing
```

#### Test Structure
- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests that verify the system as a whole
- `tests/data/`: Test data and fixtures

### Configuration Files
- `.flake8`: Flake8 configuration (excludes, line length, etc.)
- `pyproject.toml`: Configuration for various tools (black, isort, mypy, pytest)
- `.pre-commit-config.yaml`: Pre-commit hooks configuration

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Built with [MediaPipe](https://mediapipe.dev/)
- Inspired by research in computer vision and behavior analysis
