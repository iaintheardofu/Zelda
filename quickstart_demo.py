#!/usr/bin/env python3
"""
Zelda Quick Start Demo

This script demonstrates the core Zelda TDOA system without requiring
any hardware. It simulates:
- 4 receivers in a square configuration
- 1 moving emitter
- Real-time TDOA calculation and geolocation

Run with: python quickstart_demo.py
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.demo.simulator import DemoSystem
from loguru import logger

# Configure simple logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


def main():
    """Run quick start demo"""

    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗███████╗██╗     ██████╗  █████╗                    ║
║   ╚══███╔╝██╔════╝██║     ██╔══██╗██╔══██╗                   ║
║     ███╔╝ █████╗  ██║     ██║  ██║███████║                   ║
║    ███╔╝  ██╔══╝  ██║     ██║  ██║██╔══██║                   ║
║   ███████╗███████╗███████╗██████╔╝██║  ██║                   ║
║   ╚══════╝╚══════╝╚══════╝╚═════╝ ╚═╝  ╚═╝                   ║
║                                                               ║
║   Advanced TDOA Electronic Warfare Platform                  ║
║   Quick Start Demo                                           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    print("\nThis demo simulates a complete TDOA geolocation system:")
    print("  • 4 receivers positioned in a square (1000m x 1000m)")
    print("  • 1 RF emitter moving randomly within the area")
    print("  • Real-time TDOA calculation using GCC-PHAT")
    print("  • Multilateration using Taylor Series Least Squares")
    print("\nPress Ctrl+C to stop the demo.\n")

    input("Press Enter to start...")

    # Create demo system
    demo = DemoSystem(
        num_receivers=4,
        num_emitters=1,
        area_size=1000.0,
    )

    # Run for 10 iterations at 1 Hz
    try:
        demo.run(duration=10, update_rate=1.0)
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")

    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)

    print("""
Next steps:

1. Run the full demo mode:
   python -m backend.main --mode demo --num-receivers 4

2. Start the API server:
   python -m backend.main --mode api
   Then visit: http://localhost:8000/docs

3. Run tests:
   python backend/tests/test_tdoa.py

4. Explore the code:
   - Hardware abstraction: backend/core/hardware/
   - TDOA algorithms: backend/core/tdoa/
   - ML models: backend/core/ml/
   - API: backend/api/

5. Read the architecture:
   cat ARCHITECTURE.md

For more information, see README.md
    """)


if __name__ == "__main__":
    main()
