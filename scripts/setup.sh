#!/bin/bash

# Zelda Setup Script
# Sets up the development environment for Zelda TDOA system

set -e

echo "========================================="
echo "Zelda TDOA System - Setup"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found. Please install Python 3.11+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "Operating System: macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "Operating System: Linux"
else
    echo -e "${YELLOW}Warning: Unsupported OS. Attempting to continue...${NC}"
    OS="unknown"
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."

if [[ "$OS" == "macos" ]]; then
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}Homebrew not found. Please install from https://brew.sh${NC}"
        exit 1
    fi

    echo "Installing dependencies via Homebrew..."
    brew install soapysdr uhd rtl-sdr || echo -e "${YELLOW}Some packages may already be installed${NC}"

elif [[ "$OS" == "linux" ]]; then
    echo "Installing dependencies via apt (requires sudo)..."
    sudo apt-get update
    sudo apt-get install -y \
        libsoapysdr-dev \
        soapysdr-tools \
        libuhd-dev \
        uhd-host \
        rtl-sdr \
        librtlsdr-dev \
        || echo -e "${YELLOW}Some packages may already be installed${NC}"
fi

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/recordings
mkdir -p data/models
mkdir -p data/logs
mkdir -p config
mkdir -p frontend/build

echo -e "${GREEN}✓ Directories created${NC}"

# Download/prepare ML models (if any)
echo ""
echo "Preparing ML models..."
# In production, this would download pre-trained models
echo -e "${YELLOW}Using random initialization for ML models (no pre-trained models)${NC}"

# Check for SoapySDR devices
echo ""
echo "Checking for SDR devices..."
if command -v SoapySDRUtil &> /dev/null; then
    SoapySDRUtil --find || echo -e "${YELLOW}No SDR devices found (this is OK for demo mode)${NC}"
else
    echo -e "${YELLOW}SoapySDR utilities not found in PATH${NC}"
fi

# Setup complete
echo ""
echo "========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the demo:"
echo "   python -m backend.main --mode demo"
echo ""
echo "3. Or start the API server:"
echo "   python -m backend.main --mode api"
echo ""
echo "4. Run tests:"
echo "   pytest backend/tests/"
echo ""
echo "5. View the README for more information:"
echo "   cat README.md"
echo ""
echo "========================================="
