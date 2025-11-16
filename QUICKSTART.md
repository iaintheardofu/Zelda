# Zelda - Quick Start Guide

Get up and running with Zelda in 5 minutes.

## Instant Demo (No Setup Required)

The fastest way to see Zelda in action:

```bash
cd ~/zelda
python3 quickstart_demo.py
```

This will:
- Simulate 4 receivers in a 1km x 1km area
- Create a moving RF emitter
- Calculate TDOAs in real-time
- Perform multilateration to geolocate the emitter
- Show you position estimates vs. ground truth

**No SDR hardware required!**

## Installation (5 minutes)

### Prerequisites

- Python 3.11+
- macOS (Homebrew) or Linux (apt)

### Quick Install

```bash
cd ~/zelda

# Run setup script
./scripts/setup.sh

# Activate virtual environment
source venv/bin/activate

# Run tests
python backend/tests/test_tdoa.py
```

## Running Zelda

### 1. Demo Mode (Recommended for First Time)

Simulates the entire system with fake signals:

```bash
python -m backend.main --mode demo --num-receivers 4 --num-emitters 1
```

Options:
- `--num-receivers`: Number of simulated receivers (default: 4)
- `--num-emitters`: Number of simulated emitters (default: 1)
- `--duration`: Run time in seconds (default: infinite)
- `--update-rate`: Updates per second (default: 1 Hz)

### 2. API Server Mode

Start the REST API and WebSocket server:

```bash
python -m backend.main --mode api
```

Then visit:
- **API Docs**: http://localhost:8000/docs
- **Status**: http://localhost:8000/api/status
- **WebSocket**: ws://localhost:8000/ws/positions

### 3. Lab Mode (Requires SDR Hardware)

Connect real SDR hardware (RTL-SDR, USRP, KrakenSDR):

```bash
# First, check for available devices
python -c "from backend.core.hardware.soapy_backend import SoapySDRReceiver; SoapySDRReceiver.enumerate_devices()"

# Run lab mode with config file
python -m backend.main --mode lab --config config/zelda.yaml
```

## Testing Individual Components

### Test TDOA Algorithms

```python
python3 << EOF
from backend.core.tdoa.gcc_phat import gcc_phat
import numpy as np

# Generate test signals
sample_rate = 1e6
signal = np.exp(2j * np.pi * 1000 * np.arange(1024) / sample_rate)
delayed_signal = np.roll(signal, 10)

# Calculate TDOA
tdoa, confidence = gcc_phat(signal, delayed_signal, sample_rate)
print(f"TDOA: {tdoa*1e6:.2f} microseconds (confidence: {confidence:.3f})")
EOF
```

### Test Multilateration

```python
python3 << EOF
from backend.core.tdoa.multilateration import TDOAMeasurement, multilaterate_taylor_series

# Create sample measurements
measurements = [
    TDOAMeasurement((0, 0, 0), (1000, 0, 0), 1e-6, 1.0),
    TDOAMeasurement((0, 0, 0), (1000, 1000, 0), -1e-6, 1.0),
    TDOAMeasurement((0, 0, 0), (0, 1000, 0), 0.5e-6, 1.0),
]

# Solve for position
position, error = multilaterate_taylor_series(measurements)
print(f"Estimated position: {position}")
print(f"Residual error: {error:.2f}m")
EOF
```

### Test ML Signal Classifier

```python
python3 << EOF
from backend.core.ml.signal_classifier import SignalClassifier
import numpy as np

# Create classifier
classifier = SignalClassifier(signal_length=1024)

# Generate random signal
signal = np.random.randn(1024) + 1j * np.random.randn(1024)

# Classify
result = classifier.classify(signal)
print(f"Classification: {result.modulation.name}")
print(f"Confidence: {result.confidence:.3f}")
EOF
```

## API Usage Examples

### Using curl

```bash
# Get status
curl http://localhost:8000/api/status

# Add a receiver
curl -X POST http://localhost:8000/api/receivers \
  -H "Content-Type: application/json" \
  -d '{
    "receiver_id": "rx_0",
    "driver": "rtlsdr",
    "center_freq": 100e6,
    "sample_rate": 2.4e6,
    "latitude": 37.7749,
    "longitude": -122.4194,
    "altitude": 0,
    "gain": 20
  }'

# Start system
curl -X POST http://localhost:8000/api/start

# Get latest position
curl http://localhost:8000/api/positions/latest
```

### Using Python

```python
import requests
import websocket
import json

# REST API
response = requests.get("http://localhost:8000/api/status")
print(response.json())

# WebSocket for real-time updates
def on_message(ws, message):
    data = json.loads(message)
    if data.get("type") == "position_update":
        pos = data["data"]
        print(f"Position: ({pos['latitude']:.6f}, {pos['longitude']:.6f})")

ws = websocket.WebSocketApp(
    "ws://localhost:8000/ws/positions",
    on_message=on_message
)
ws.run_forever()
```

## Project Structure

```
zelda/
├── ARCHITECTURE.md       # Detailed system architecture
├── README.md            # Project overview
├── QUICKSTART.md        # This file
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Docker orchestration
├── quickstart_demo.py   # Instant demo script
│
├── backend/
│   ├── main.py          # Main entry point
│   ├── api/             # REST API & WebSocket
│   ├── core/
│   │   ├── hardware/    # SDR hardware abstraction
│   │   ├── tdoa/        # TDOA algorithms
│   │   ├── ml/          # Machine learning
│   │   └── tracking/    # Kalman filters
│   ├── demo/            # Simulation system
│   └── tests/           # Unit tests
│
├── frontend/            # React web interface (TBD)
├── config/              # Configuration files
├── data/                # Data storage
│   ├── recordings/      # Signal recordings
│   ├── models/          # ML models
│   └── logs/            # Logs
└── scripts/             # Utility scripts
    └── setup.sh         # Setup automation
```

## Troubleshooting

### "SoapySDR not found"

```bash
# macOS
brew install soapysdr

# Linux
sudo apt-get install libsoapysdr-dev soapysdr-tools
```

### "PyTorch not available"

ML features will be disabled. To enable:

```bash
pip install torch torchvision
```

### "No SDR devices found"

This is normal if you don't have hardware. Use demo mode:

```bash
python -m backend.main --mode demo
```

### "Import errors"

Make sure you're in the virtual environment:

```bash
source venv/bin/activate
```

## Next Steps

1. **Explore the code**: Start with `backend/demo/simulator.py`
2. **Read the architecture**: `ARCHITECTURE.md`
3. **Try real hardware**: Connect an RTL-SDR and run lab mode
4. **Train ML models**: See `docs/ml_training.md` (coming soon)
5. **Build the frontend**: React + Three.js visualization
6. **Deploy**: Use `docker-compose up` for full stack

## Getting Help

- **Issues**: https://github.com/yourusername/zelda/issues
- **Discussions**: https://github.com/yourusername/zelda/discussions
- **Documentation**: See `docs/` directory

## What's Working

✅ Hardware abstraction (SoapySDR)
✅ GCC-PHAT TDOA calculation
✅ Taylor Series multilateration
✅ Genetic algorithm optimization
✅ Signal classification (CNN/ResNet)
✅ Kalman filter tracking
✅ REST API & WebSocket
✅ Simulation & demo mode
✅ Unit tests

## What's Coming

⏳ Web frontend (React + Three.js)
⏳ Real-time visualization
⏳ Distributed receiver coordination
⏳ Pre-trained ML models
⏳ Cloud deployment
⏳ Mobile app

---

**Zelda** - Advanced TDOA Electronic Warfare Platform
*Making the Invisible, Visible*
