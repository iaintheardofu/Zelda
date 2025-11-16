"""
Zelda - Main Application Entry Point

Runs the TDOA system in various modes:
- Demo mode (simulated signals)
- Lab mode (real hardware, local)
- Field mode (distributed receivers)
"""

import argparse
import asyncio
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)


def run_demo_mode(args):
    """Run Zelda in demo mode with simulated signals"""
    logger.info("Starting Zelda in DEMO mode")

    from .demo.simulator import DemoSystem

    demo = DemoSystem(
        num_receivers=args.num_receivers,
        num_emitters=args.num_emitters,
    )

    demo.run(
        duration=args.duration,
        update_rate=args.update_rate,
    )


def run_lab_mode(args):
    """Run Zelda in lab mode with real hardware"""
    logger.info("Starting Zelda in LAB mode")

    from .systems.lab_system import LabSystem

    system = LabSystem(config_file=args.config)
    system.run()


def run_field_mode(args):
    """Run Zelda in field mode (distributed)"""
    logger.info("Starting Zelda in FIELD mode")

    from .systems.field_system import FieldSystem

    system = FieldSystem(config_file=args.config)
    system.run()


def run_api_server(args):
    """Run the API server"""
    logger.info("Starting Zelda API server")

    import uvicorn
    from .api.app import create_app

    app = create_app()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Zelda - Advanced TDOA Electronic Warfare Platform"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "lab", "field", "api"],
        default="demo",
        help="Operating mode"
    )

    # Demo mode options
    parser.add_argument(
        "--num-receivers",
        type=int,
        default=4,
        help="Number of receivers (demo mode)"
    )

    parser.add_argument(
        "--num-emitters",
        type=int,
        default=1,
        help="Number of emitters (demo mode)"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Demo duration in seconds (None = infinite)"
    )

    parser.add_argument(
        "--update-rate",
        type=float,
        default=1.0,
        help="Update rate in Hz (demo mode)"
    )

    # Lab/Field mode options
    parser.add_argument(
        "--config",
        type=str,
        default="config/zelda.yaml",
        help="Configuration file path"
    )

    # API server options
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API server host"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Set log level
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Run appropriate mode
    try:
        if args.mode == "demo":
            run_demo_mode(args)
        elif args.mode == "lab":
            run_lab_mode(args)
        elif args.mode == "field":
            run_field_mode(args)
        elif args.mode == "api":
            run_api_server(args)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
