# Zelda - Advanced TDOA Electronic Warfare Platform

## Vision
Zelda is a unified Time Difference of Arrival (TDOA) geolocation system that combines multi-platform SDR hardware support, machine learning-enhanced signal processing, and real-time visualization to provide capabilities that don't exist in current open-source EW tools.

## Core Innovations

1. **Hardware Agnostic**: Unified API across KrakenSDR, USRP, RTL-SDR, and other SoapySDR-compatible devices
2. **ML-Enhanced**: Neural networks for signal classification, interference rejection, and pattern recognition
3. **Advanced Algorithms**: Multiple TDOA/multilateration methods with genetic algorithm optimization
4. **Real-Time**: Sub-second geolocation with live tracking and visualization
5. **Production Ready**: Web-based interface for demonstrations and operational use

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Zelda Platform                          │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Next.js + React + Lovable Cloud)                 │
│  ├── Cyberpunk Dashboard (deployed via Lovable → Vercel)   │
│  ├── Real-Time Spectrum Waterfall (Canvas-based)            │
│  ├── Custom Analytics Charts (SVG, no Grafana)              │
│  ├── 3D Map Visualization (Three.js)                        │
│  └── WebSocket Real-Time Updates                            │
├─────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI + WebSocket)                            │
│  ├── REST API for configuration                             │
│  ├── WebSocket for real-time data streaming                 │
│  ├── Control interface for SDR parameters                   │
│  └── CORS support for Lovable/Vercel deployment             │
├─────────────────────────────────────────────────────────────┤
│  Processing Pipeline                                        │
│  ├── Signal Acquisition (multi-threaded)                    │
│  ├── ML Classification (PyTorch/TensorFlow)                 │
│  ├── TDOA Calculation (NumPy/SciPy)                         │
│  ├── Geolocation (optimization algorithms)                  │
│  └── Tracking (Kalman filtering)                            │
├─────────────────────────────────────────────────────────────┤
│  ML/AI Engine                                               │
│  ├── Signal Classifier (ResNet-based)                       │
│  ├── Interference Detector                                  │
│  ├── Automatic Modulation Classification                    │
│  └── Anomaly Detection                                      │
├─────────────────────────────────────────────────────────────┤
│  TDOA Core                                                   │
│  ├── GCC-PHAT (Generalized Cross-Correlation)               │
│  ├── Multilateration Solver                                 │
│  │   ├── Taylor Series Least Squares                        │
│  │   ├── Genetic Algorithm Optimization                     │
│  │   └── Hybrid ML-assisted solver                          │
│  └── Position Tracking (EKF/UKF)                            │
├─────────────────────────────────────────────────────────────┤
│  Hardware Abstraction Layer (SoapySDR)                      │
│  ├── KrakenSDR Driver                                       │
│  ├── USRP Driver (UHD)                                      │
│  ├── RTL-SDR Driver                                         │
│  └── Phase Coherence Calibration                            │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                  │
│  ├── Supabase (PostgreSQL + Realtime + Auth + Storage)     │
│  ├── InfluxDB (optional - time-series metrics)              │
│  └── Redis (optional - real-time cache)                     │
└─────────────────────────────────────────────────────────────┘
```

## Technical Stack

### Core Languages
- **Python 3.11+**: Primary backend language
- **C++**: Performance-critical DSP kernels
- **JavaScript/TypeScript**: Frontend

### SDR & Signal Processing
- **SoapySDR**: Hardware abstraction
- **NumPy/SciPy**: Numerical computation
- **GNU Radio** (optional): Advanced signal processing chains

### Machine Learning
- **PyTorch**: Primary ML framework
- **TensorFlow/Keras**: Alternative models
- **ONNX**: Model interchange
- **scikit-learn**: Classical ML algorithms

### Optimization
- **DEAP**: Genetic algorithms
- **SciPy Optimize**: Non-linear solvers
- **CVXPY**: Convex optimization

### Quantum (Exploratory)
- **Qiskit**: Quantum-inspired optimization
- **Cirq**: Alternative quantum framework

### Backend Services
- **FastAPI**: REST API server
- **uvicorn**: ASGI server
- **WebSocket**: Real-time communication
- **Celery**: Distributed task queue

### Data Storage
- **Supabase**: PostgreSQL + Realtime + Auth + Storage (primary)
- **InfluxDB**: Time-series data (optional)
- **Redis**: Caching and pub/sub (optional)

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Lovable Cloud**: Deployment platform → Vercel hosting
- **Custom SVG/Canvas Charts**: No external chart libraries
- **Three.js**: 3D visualization
- **Mapbox GL**: Mapping
- **TailwindCSS**: Cyberpunk theming with custom animations

### DevOps & Deployment
- **Lovable Cloud**: Frontend deployment via GitHub integration
- **Vercel**: Auto-scaling frontend hosting
- **Docker**: Backend containerization
- **docker-compose**: Local development environment
- **pytest**: Testing
- **GitHub Actions**: CI/CD
- **Railway/Render**: Python backend hosting options

## Key Algorithms

### 1. TDOA Calculation
- **GCC-PHAT** (primary): Robust to multipath, computationally efficient
- **Cross-correlation**: Baseline method
- **Adaptive filtering**: Noise reduction

### 2. Multilateration
- **Taylor Series Least Squares**: Fast convergence
- **Genetic Algorithm**: Global optimization for difficult geometries
- **Hybrid ML-NN**: Neural network-assisted position estimation
- **Weighted Least Squares**: Error-weighted solution

### 3. Tracking
- **Extended Kalman Filter (EKF)**: Non-linear tracking
- **Unscented Kalman Filter (UKF)**: Better for high non-linearity
- **Particle Filter**: Multi-hypothesis tracking

### 4. Signal Classification (ML)
- **ResNet-18**: Deep residual network for signal classification
- **CNN**: Convolutional neural network for I/Q data
- **Attention Mechanisms**: Focus on relevant signal features
- **Transfer Learning**: Pre-trained models for common modulations

## Data Flow

```
SDR Hardware → SoapySDR → Phase Calibration → I/Q Samples
                                                    ↓
                                          ML Signal Classifier
                                                    ↓
                                         Filter & Preprocessing
                                                    ↓
                                          GCC-PHAT (TDOA)
                                                    ↓
                                         Multilateration Solver
                                                    ↓
                                            Kalman Filter
                                                    ↓
                                          Geolocation Result
                                                    ↓
                                     WebSocket → Frontend Display
                                                    ↓
                                            InfluxDB Storage
```

## Deployment Modes

### 1. Development Mode (Local)
- Single receiver simulation
- Pre-recorded I/Q files
- No hardware required
- Frontend: `npm run dev` (localhost:3000)
- Backend: `uvicorn api.app:app --reload` (localhost:8000)

### 2. Lab Mode (Local Multi-SDR)
- Multiple SDRs on same machine
- Simulated RF environment
- Full stack running locally via docker-compose

### 3. Production Mode (Lovable Cloud)
- **Frontend**: Deployed to Vercel via Lovable
- **Backend**: Railway/Render/DigitalOcean
- **Database**: Supabase (managed PostgreSQL)
- **Real-time**: WebSocket streaming
- **Monitoring**: Built-in React analytics dashboard

### 4. Field Mode (Distributed)
- Distributed receivers (GPS-synchronized)
- Network coordination
- Cloud backend processing
- Lovable Cloud frontend for monitoring

### 5. Demo Mode (Public)
- Web-based interface at https://zelda.vercel.app
- Simulated emitters and real-time data
- Interactive cyberpunk visualization
- No hardware required

## Performance Targets

- **Latency**: < 100ms from signal reception to geolocation
- **Accuracy**: < 10m CEP (Circular Error Probable) at 1km range
- **Throughput**: 100+ TDOA calculations per second
- **Scalability**: Support 10+ simultaneous emitters
- **Receiver Count**: 3-16 coherent receivers

## Security Considerations

- **Authentication**: JWT-based API access
- **Encryption**: TLS for all network communication
- **Isolation**: Containerized services
- **Logging**: Comprehensive audit trail
- **Rate Limiting**: DDoS protection

## Future Enhancements

1. **Quantum Computing**: Explore QAOA for multilateration optimization
2. **Federated Learning**: Distributed ML model training
3. **5G/WiFi**: Support for modern wireless protocols
4. **Swarm Coordination**: Multiple drone-mounted receivers
5. **Adversarial ML**: Robust signal classification under jamming

## License & Attribution

Open-source (MIT License) with attribution to:
- SoapySDR project
- KrakenRF community
- Ettus Research (UHD)
- GNU Radio community
- ML/RF research community

---

**Project Zelda** - Advanced TDOA Electronic Warfare Platform
*Making the invisible, visible.*
