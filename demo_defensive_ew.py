#!/usr/bin/env python3
"""
ZELDA DEFENSIVE EW SUITE - COMPREHENSIVE DEMONSTRATION

Demonstrates all defensive electronic warfare capabilities:
1. Jamming Detection & Characterization
2. Spoofing Detection (GPS, Cellular, WiFi)
3. Anti-Jam Signal Processing
4. Signal Simulation (testing only)

ALL CAPABILITIES ARE DEFENSIVE - Detection, analysis, and mitigation only.
NO RF TRANSMISSION occurs - all demonstrations use software simulation.

Legal Use: Testing, education, authorized security research
"""

import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.ew.jamming_detection import AdaptiveJammingDetector, JammingType
from backend.core.ew.spoofing_detection import IntegratedSpoofingDetector
from backend.core.ew.antijam_processing import AdaptiveAntiJamProcessor, generate_antijam_report
from backend.core.ew.signal_simulator import RFSignalSimulator, SignalType, JammingSimulationType

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     ███████╗███████╗██╗     ██████╗  █████╗     ███████╗██╗    ██╗  ║
║     ╚══███╔╝██╔════╝██║     ██╔══██╗██╔══██╗    ██╔════╝██║    ██║  ║
║       ███╔╝ █████╗  ██║     ██║  ██║███████║    █████╗  ██║ █╗ ██║  ║
║      ███╔╝  ██╔══╝  ██║     ██║  ██║██╔══██║    ██╔══╝  ██║███╗██║  ║
║     ███████╗███████╗███████╗██████╔╝██║  ██║    ███████╗╚███╔███╔╝  ║
║     ╚══════╝╚══════╝╚══════╝╚═════╝ ╚═╝  ╚═╝    ╚══════╝ ╚══╝╚══╝   ║
║                                                                      ║
║              DEFENSIVE ELECTRONIC WARFARE SUITE                      ║
║                    Detection • Analysis • Mitigation                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

   🛡️  100% DEFENSIVE CAPABILITIES - NO RF TRANSMISSION  🛡️

   Legal Use: Security monitoring, threat detection, education
"""


def demo_jamming_detection():
    """Demonstrate jamming detection capabilities"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 1: JAMMING DETECTION & CHARACTERIZATION")
    print("=" * 70)

    # Create detector
    detector = AdaptiveJammingDetector(sample_rate=40e6, window_size=4096)

    # Create simulator
    simulator = RFSignalSimulator(sample_rate=40e6)

    # Test various jamming types
    jamming_scenarios = [
        ("Clean Signal (No Jamming)", JammingSimulationType.NONE, 0),
        ("Barrage Jamming (Wideband Noise)", JammingSimulationType.BARRAGE, 15),
        ("Spot Jamming (Narrowband CW)", JammingSimulationType.SPOT, 12),
        ("Pulse Jamming (30% Duty Cycle)", JammingSimulationType.PULSE, 18),
        ("Swept Jamming (Frequency Hopping)", JammingSimulationType.SWEPT, 10),
    ]

    for scenario_name, jamming_type, jammer_power_db in jamming_scenarios:
        print(f"\n--- {scenario_name} ---")

        # Generate test signal
        if jamming_type == JammingSimulationType.NONE:
            signal = simulator.generate_clean_signal(
                SignalType.QPSK, duration_sec=0.0001, carrier_freq=1e9, snr_db=20
            )
        else:
            signal, _ = simulator.generate_jammed_signal(
                SignalType.QPSK, duration_sec=0.0001, carrier_freq=1e9,
                jamming_type=jamming_type, jammer_power_db=jammer_power_db
            )

        # Detect jamming
        result = detector.detect(signal)

        # Print results
        status = "⚠️  JAMMED" if result.is_jammed else "✓  CLEAN"
        print(f"Status: {status}")

        if result.is_jammed:
            print(f"Type: {result.jamming_type.value.upper()}")
            print(f"Confidence: {result.confidence*100:.1f}%")

        print(f"SNR: {result.signal_to_noise_db:+.2f} dB")
        print(f"Interference Power: {result.interference_power_db:+.2f} dB")
        print(f"Affected Bandwidth: {result.affected_bandwidth_hz/1e6:.2f} MHz")

        if result.characteristics:
            print(f"Characteristics:")
            for key, value in list(result.characteristics.items())[:3]:
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")


def demo_spoofing_detection():
    """Demonstrate spoofing detection capabilities"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 2: SPOOFING DETECTION (GPS, Cellular, WiFi)")
    print("=" * 70)

    # Create integrated detector
    detector = IntegratedSpoofingDetector()

    # Test Case 1: GPS Spoofing
    print("\n--- GPS Spoofing Detection ---")
    simulator = RFSignalSimulator(sample_rate=10e6)
    # Simulate overly strong GPS signal (spoofing indicator)
    gps_signal = simulator.generate_clean_signal(
        SignalType.GPS_L1, duration_sec=0.001, carrier_freq=1575.42e6, amplitude=5.0
    )

    # Test Case 2: Cellular IMSI Catcher
    print("\n--- Cellular IMSI Catcher Detection ---")
    cell_info = {
        'cell_id': 99999,  # Unknown cell
        'lac': 12345,
        'mcc': 310,
        'mnc': 260,
        'signal_strength_dbm': -40,  # Very strong (suspicious)
        'network_type': '2G',  # Downgrade from 4G
        'encryption': False  # No encryption (major red flag)
    }

    # Test Case 3: WiFi Evil Twin
    print("\n--- WiFi Evil Twin Detection ---")
    wifi_aps = [
        {
            'ssid': 'Starbucks WiFi',
            'bssid': '00:11:22:33:44:55',  # Fake MAC
            'signal_strength_dbm': -30,  # Very strong
            'channel': 6,
            'encryption': 'Open'  # Legitimate Starbucks uses WPA2
        },
        {
            'ssid': 'Free Airport WiFi',
            'bssid': 'AA:BB:CC:DD:EE:FF',
            'signal_strength_dbm': -35,
            'channel': 11,
            'encryption': 'Open'  # Common evil twin target
        }
    ]

    # Run all detections
    results = detector.detect_all(
        gps_signal=gps_signal,
        cell_info=cell_info,
        wifi_aps=wifi_aps
    )

    # Print comprehensive report
    report = detector.generate_summary_report(results)
    print("\n" + report)


def demo_antijam_processing():
    """Demonstrate anti-jam signal processing"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 3: ANTI-JAM SIGNAL PROCESSING")
    print("=" * 70)

    # Create processor
    processor = AdaptiveAntiJamProcessor(sample_rate=40e6)

    # Create simulator
    simulator = RFSignalSimulator(sample_rate=40e6)

    # Test anti-jam methods
    test_scenarios = [
        ("Spot Jamming → Notch Filter", JammingSimulationType.SPOT, 15),
        ("Barrage Jamming → Whitening Filter", JammingSimulationType.BARRAGE, 12),
        ("Pulse Jamming → Pulse Blanking", JammingSimulationType.PULSE, 20),
        ("Swept Jamming → Spectral Excision", JammingSimulationType.SWEPT, 10),
    ]

    for scenario_name, jamming_type, jammer_power_db in test_scenarios:
        print(f"\n--- {scenario_name} ---")

        # Generate jammed signal
        jammed_signal, clean_signal = simulator.generate_jammed_signal(
            SignalType.QPSK, duration_sec=0.0001, carrier_freq=1e9,
            jamming_type=jamming_type, jammer_power_db=jammer_power_db
        )

        # Calculate original SNR
        jammer = jammed_signal - clean_signal
        original_snr = 10 * np.log10(
            np.mean(np.abs(clean_signal)**2) / (np.mean(np.abs(jammer)**2) + 1e-12)
        )

        # Apply anti-jam processing
        result = processor.process(jammed_signal)

        # Calculate recovered SNR (approximate)
        recovered_snr = original_snr + result.snr_improvement_db

        # Print results
        print(f"Method: {result.method_used}")
        print(f"Original SNR: {original_snr:+.2f} dB")
        print(f"Recovered SNR: {recovered_snr:+.2f} dB")
        print(f"Improvement: {result.snr_improvement_db:+.2f} dB")
        print(f"Interference Suppressed: {result.interference_suppressed_db:+.2f} dB")
        print(f"Status: {'✓ SUCCESS' if result.success else '✗ LIMITED EFFECT'}")


def demo_integrated_workflow():
    """Demonstrate complete detection → mitigation workflow"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 4: INTEGRATED WORKFLOW (Detect → Mitigate)")
    print("=" * 70)

    # Create all systems
    simulator = RFSignalSimulator(sample_rate=40e6)
    jammer_detector = AdaptiveJammingDetector(sample_rate=40e6)
    antijam_processor = AdaptiveAntiJamProcessor(sample_rate=40e6)

    print("\n--- Integrated Workflow Example ---")
    print("Scenario: Communications link under barrage jamming attack\n")

    # Step 1: Generate jammed signal
    print("STEP 1: Signal Reception")
    jammed_signal, clean_signal = simulator.generate_jammed_signal(
        SignalType.QPSK, duration_sec=0.0001, carrier_freq=1e9,
        jamming_type=JammingSimulationType.BARRAGE, jammer_power_db=15
    )
    print("  Received signal with suspected interference")

    # Step 2: Detect jamming
    print("\nSTEP 2: Jamming Detection")
    detection_result = jammer_detector.detect(jammed_signal)

    if detection_result.is_jammed:
        print(f"  ⚠️  JAMMING DETECTED")
        print(f"  Type: {detection_result.jamming_type.value.upper()}")
        print(f"  Confidence: {detection_result.confidence*100:.1f}%")
        print(f"  SNR: {detection_result.signal_to_noise_db:+.2f} dB")

        # Step 3: Apply appropriate mitigation
        print("\nSTEP 3: Applying Countermeasures")

        # Map jamming type to mitigation hint
        jamming_type_map = {
            JammingType.BARRAGE: "barrage",
            JammingType.SPOT: "spot",
            JammingType.PULSE: "pulse",
            JammingType.SWEPT: "swept"
        }
        mitigation_hint = jamming_type_map.get(detection_result.jamming_type, None)

        # Apply mitigation
        mitigation_result = antijam_processor.process(jammed_signal, jamming_type=mitigation_hint)

        print(f"  Method: {mitigation_result.method_used}")
        print(f"  SNR Improvement: {mitigation_result.snr_improvement_db:+.2f} dB")
        print(f"  Interference Suppression: {mitigation_result.interference_suppressed_db:+.2f} dB")

        # Step 4: Verify results
        print("\nSTEP 4: Verification")
        cleaned_signal = mitigation_result.cleaned_signal
        final_detection = jammer_detector.detect(cleaned_signal)

        if final_detection.signal_to_noise_db > detection_result.signal_to_noise_db:
            print(f"  ✓ Signal quality improved:")
            print(f"    Before: {detection_result.signal_to_noise_db:+.2f} dB SNR")
            print(f"    After:  {final_detection.signal_to_noise_db:+.2f} dB SNR")
            print("  ✓ Communications link restored")
        else:
            print("  ⚠  Additional mitigation may be required")

    else:
        print("  ✓ No jamming detected, signal is clean")


def print_capabilities_summary():
    """Print summary of all defensive capabilities"""
    print("\n" + "=" * 70)
    print("ZELDA DEFENSIVE EW SUITE - CAPABILITIES SUMMARY")
    print("=" * 70)

    capabilities = {
        "Jamming Detection": [
            "✓ Barrage jamming (wideband noise)",
            "✓ Spot jamming (narrowband CW)",
            "✓ Swept jamming (frequency hopping)",
            "✓ Pulse jamming (on/off bursts)",
            "✓ Deceptive jamming (signal mimicking)",
            "✓ Adaptive baseline learning",
            "✓ Real-time characterization"
        ],
        "Spoofing Detection": [
            "✓ GPS spoofing (meaconing & simulation)",
            "✓ Cellular IMSI catchers",
            "✓ Rogue femtocells",
            "✓ WiFi evil twin attacks",
            "✓ Rogue access points",
            "✓ Multi-system correlation"
        ],
        "Anti-Jam Processing": [
            "✓ Adaptive notch filtering",
            "✓ Spectral excision",
            "✓ Adaptive whitening",
            "✓ Pulse blanking",
            "✓ Automatic method selection",
            "✓ Cascade processing"
        ],
        "Signal Simulation": [
            "✓ QPSK, QAM, OFDM modulations",
            "✓ GPS L1 simulation",
            "✓ All jamming types",
            "✓ Comprehensive test suites",
            "✓ SOFTWARE ONLY - NO TRANSMISSION"
        ]
    }

    for category, items in capabilities.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")

    print("\n" + "=" * 70)
    print("🛡️  ALL CAPABILITIES ARE DEFENSIVE")
    print("🛡️  NO RF TRANSMISSION - Detection & Analysis Only")
    print("🛡️  Legal for: Security monitoring, threat detection, education")
    print("=" * 70)


def main():
    """Run comprehensive defensive EW demonstration"""
    print(BANNER)

    try:
        # Run all demonstrations
        demo_jamming_detection()
        demo_spoofing_detection()
        demo_antijam_processing()
        demo_integrated_workflow()

        # Print capabilities summary
        print_capabilities_summary()

        print("\n" + "=" * 70)
        print("✓ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print("\n📁 Defensive EW Modules:")
        print("  • backend/core/ew/jamming_detection.py")
        print("  • backend/core/ew/spoofing_detection.py")
        print("  • backend/core/ew/antijam_processing.py")
        print("  • backend/core/ew/signal_simulator.py")

        print("\n📚 Usage:")
        print("  python3 demo_defensive_ew.py")

        print("\n⚖️  Legal Notice:")
        print("  All demonstrations use software simulation only.")
        print("  No RF transmission occurs.")
        print("  For authorized security research and education.")

        print("\n🚀 Next Steps:")
        print("  1. Integrate with ZELDA main platform")
        print("  2. Add real-time visualization dashboard")
        print("  3. Deploy in production security monitoring")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
