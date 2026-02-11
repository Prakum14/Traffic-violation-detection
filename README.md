# Traffic-violation-detection
In this project the tool can detect the vehicles violating the traffic rules by playing a traffic video

# Traffic Violation Detection System

AI-powered traffic violation detection using YOLOv8 for vehicle tracking and helmet detection.

## Features

- ✅ Detects no-helmet violations
- ✅ Detects wrong-way driving
- ✅ Supports curved road dividers
- ✅ Real-time video processing
- ✅ Saves violation snapshots

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/traffic-violation-detector.git
cd traffic-violation-detector
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your API key:
- Copy `config.example.py` to `config.py`
- Add your Roboflow API key

## Usage

1. Calibrate road divider:
```bash
python calibrate_curved_divider.py
```

2. Update `DIVIDER_POINTS` in `traffic_detector_refined.py`

3. Run detection:
```bash
python traffic_detector_refined.py
```

## Configuration

See `REFINED_VERSION_GUIDE.md` for detailed configuration instructions.

## Requirements

- Python 3.8+
- OpenCV
- YOLOv8 (ultralytics)
- NumPy

## License

MIT License
