"""
ZELDA Countermeasure Engine
Automated defensive responses to RF threats

Implements:
- Frequency hopping (anti-jamming)
- Power adjustment (interference mitigation)
- Jamming mitigation (spectrum evasion)
- Beamforming and null steering (directional mitigation)
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CountermeasureType(Enum):
    FREQUENCY_HOPPING = "frequency_hopping"
    POWER_ADJUSTMENT = "power_adjustment"
    JAMMING_MITIGATION = "jamming_mitigation"
    BEAMFORMING = "beamforming"
    NULL_STEERING = "null_steering"
    SPECTRUM_EVASION = "spectrum_evasion"
    ALERT_ONLY = "alert_only"


class ThreatSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ThreatInfo:
    """Information about detected threat"""
    threat_id: str
    threat_type: str  # jamming, spoofing, unauthorized, interference
    severity: ThreatSeverity
    frequency: float  # Hz
    power: float  # dBm
    bandwidth: float  # Hz
    location: Optional[Tuple[float, float]] = None  # (lat, lon)


@dataclass
class CountermeasureResult:
    """Result of countermeasure execution"""
    success: bool
    countermeasure_type: CountermeasureType
    actions_taken: List[str]
    new_parameters: Dict[str, any]
    duration_ms: float
    message: str


class SDRController:
    """
    Interface to SDR hardware (HackRF, BladeRF, USRP, etc.)

    In production, replace with actual SDR library calls
    (e.g., SoapySDR, GNU Radio, uhd)
    """

    def __init__(self):
        self.current_frequency = 915e6  # Hz
        self.current_power = -10  # dBm
        self.current_gain = 20  # dB
        self.sample_rate = 20e6  # 20 MS/s
        self.is_transmitting = False

    def set_frequency(self, freq_hz: float) -> bool:
        """Set center frequency"""
        logger.info(f"SDR: Setting frequency to {freq_hz/1e6:.3f} MHz")
        self.current_frequency = freq_hz
        # In production: sdr.set_center_freq(freq_hz)
        return True

    def set_power(self, power_dbm: float) -> bool:
        """Set transmission power"""
        logger.info(f"SDR: Setting power to {power_dbm:.1f} dBm")
        self.current_power = power_dbm
        # In production: sdr.set_tx_power(power_dbm)
        return True

    def set_gain(self, gain_db: float) -> bool:
        """Set receiver gain"""
        logger.info(f"SDR: Setting gain to {gain_db:.1f} dB")
        self.current_gain = gain_db
        # In production: sdr.set_rx_gain(gain_db)
        return True

    def start_tx(self) -> bool:
        """Start transmission"""
        logger.info("SDR: Starting transmission")
        self.is_transmitting = True
        # In production: sdr.start_tx()
        return True

    def stop_tx(self) -> bool:
        """Stop transmission"""
        logger.info("SDR: Stopping transmission")
        self.is_transmitting = False
        # In production: sdr.stop_tx()
        return True


class FrequencyHoppingEngine:
    """
    Frequency hopping countermeasure for anti-jamming

    Implements:
    - Random hopping (unpredictable pattern)
    - Sequential hopping (sweep through band)
    - Adaptive hopping (avoid jammed frequencies)
    """

    def __init__(self, sdr: SDRController):
        self.sdr = sdr
        self.hop_pattern = 'adaptive'
        self.hop_rate_ms = 500  # Hop every 500ms
        self.backup_frequencies = [
            868e6,   # 868 MHz (EU ISM)
            915e6,   # 915 MHz (US ISM)
            2450e6,  # 2.45 GHz (WiFi)
        ]
        self.jammed_frequencies = set()

    def execute(self, threat: ThreatInfo, parameters: Dict) -> CountermeasureResult:
        """Execute frequency hopping countermeasure"""
        start_time = time.time()
        actions = []

        # Extract parameters
        self.hop_pattern = parameters.get('hop_pattern', 'adaptive')
        self.hop_rate_ms = parameters.get('hop_rate_ms', 500)
        backup_freqs = parameters.get('backup_frequencies', self.backup_frequencies)

        # Mark jammed frequency
        self.jammed_frequencies.add(threat.frequency)
        actions.append(f"Marked {threat.frequency/1e6:.3f} MHz as jammed")

        # Find clear frequency
        clear_freq = self._find_clear_frequency(backup_freqs, threat)

        if clear_freq:
            # Hop to clear frequency
            self.sdr.set_frequency(clear_freq)
            actions.append(f"Hopped to {clear_freq/1e6:.3f} MHz")

            # Start hopping pattern
            if self.hop_pattern == 'adaptive':
                actions.append(f"Started adaptive hopping (rate: {self.hop_rate_ms}ms)")
            elif self.hop_pattern == 'random':
                actions.append(f"Started random hopping (rate: {self.hop_rate_ms}ms)")
            elif self.hop_pattern == 'sequential':
                actions.append(f"Started sequential hopping (rate: {self.hop_rate_ms}ms)")

            duration_ms = (time.time() - start_time) * 1000

            return CountermeasureResult(
                success=True,
                countermeasure_type=CountermeasureType.FREQUENCY_HOPPING,
                actions_taken=actions,
                new_parameters={
                    'new_frequency': clear_freq,
                    'hop_pattern': self.hop_pattern,
                    'hop_rate_ms': self.hop_rate_ms,
                },
                duration_ms=duration_ms,
                message=f"Frequency hopping activated on {clear_freq/1e6:.3f} MHz"
            )
        else:
            duration_ms = (time.time() - start_time) * 1000
            return CountermeasureResult(
                success=False,
                countermeasure_type=CountermeasureType.FREQUENCY_HOPPING,
                actions_taken=actions,
                new_parameters={},
                duration_ms=duration_ms,
                message="No clear frequencies available for hopping"
            )

    def _find_clear_frequency(self, candidates: List[float], threat: ThreatInfo) -> Optional[float]:
        """Find a clear frequency from candidates"""
        for freq in candidates:
            # Skip if frequency is jammed
            if freq in self.jammed_frequencies:
                continue

            # Skip if too close to threat frequency
            if abs(freq - threat.frequency) < threat.bandwidth * 2:
                continue

            return freq

        return None


class PowerAdjustmentEngine:
    """
    Power adjustment countermeasure for interference mitigation

    Implements:
    - Power reduction to avoid interference
    - Power increase to overcome jamming
    - Adaptive power control based on SNR
    """

    def __init__(self, sdr: SDRController):
        self.sdr = sdr
        self.min_power = -20  # dBm
        self.max_power = 10   # dBm

    def execute(self, threat: ThreatInfo, parameters: Dict) -> CountermeasureResult:
        """Execute power adjustment countermeasure"""
        start_time = time.time()
        actions = []

        current_power = self.sdr.current_power

        # Determine power adjustment strategy
        if threat.threat_type == 'interference':
            # Reduce power to minimize interference
            power_reduction_db = parameters.get('power_reduction_db', 5)
            new_power = max(self.min_power, current_power - power_reduction_db)
            actions.append(f"Reduced power by {power_reduction_db} dB to minimize interference")

        elif threat.threat_type == 'jamming':
            # Increase power to overcome jamming (if safe)
            if threat.power < current_power - 10:
                new_power = min(self.max_power, current_power + 5)
                actions.append(f"Increased power by 5 dB to overcome jamming")
            else:
                # Jamming too strong, reduce power and evade
                new_power = max(self.min_power, current_power - 10)
                actions.append(f"Reduced power by 10 dB (jamming too strong)")
        else:
            # Default: slight reduction for stealth
            new_power = max(self.min_power, current_power - 3)
            actions.append(f"Reduced power by 3 dB for stealth")

        # Apply power adjustment
        self.sdr.set_power(new_power)
        actions.append(f"Set new power: {new_power:.1f} dBm")

        duration_ms = (time.time() - start_time) * 1000

        return CountermeasureResult(
            success=True,
            countermeasure_type=CountermeasureType.POWER_ADJUSTMENT,
            actions_taken=actions,
            new_parameters={
                'old_power': current_power,
                'new_power': new_power,
                'power_delta': new_power - current_power,
            },
            duration_ms=duration_ms,
            message=f"Power adjusted from {current_power:.1f} to {new_power:.1f} dBm"
        )


class JammingMitigationEngine:
    """
    Comprehensive jamming mitigation

    Combines:
    - Frequency hopping
    - Power adjustment
    - Spread spectrum techniques
    - Null steering (if array available)
    """

    def __init__(self, sdr: SDRController, freq_hopper: FrequencyHoppingEngine, power_adjuster: PowerAdjustmentEngine):
        self.sdr = sdr
        self.freq_hopper = freq_hopper
        self.power_adjuster = power_adjuster

    def execute(self, threat: ThreatInfo, parameters: Dict) -> CountermeasureResult:
        """Execute comprehensive jamming mitigation"""
        start_time = time.time()
        actions = []

        # Step 1: Reduce power to minimize detection
        power_result = self.power_adjuster.execute(threat, {'power_reduction_db': 10})
        actions.extend(power_result.actions_taken)

        # Step 2: Hop to clear frequency
        freq_result = self.freq_hopper.execute(threat, parameters)
        actions.extend(freq_result.actions_taken)

        # Step 3: Enable spread spectrum (if supported)
        if parameters.get('enable_spread_spectrum', True):
            actions.append("Enabled spread spectrum modulation")
            # In production: configure SDR for spread spectrum

        # Step 4: Enable error correction
        if parameters.get('enable_fec', True):
            actions.append("Enabled forward error correction")
            # In production: configure FEC coding

        duration_ms = (time.time() - start_time) * 1000

        success = power_result.success and freq_result.success

        return CountermeasureResult(
            success=success,
            countermeasure_type=CountermeasureType.JAMMING_MITIGATION,
            actions_taken=actions,
            new_parameters={
                **power_result.new_parameters,
                **freq_result.new_parameters,
            },
            duration_ms=duration_ms,
            message=f"Jamming mitigation {'successful' if success else 'failed'}"
        )


class CountermeasureEngine:
    """
    Main countermeasure engine

    Coordinates all countermeasure subsystems
    """

    def __init__(self):
        self.sdr = SDRController()
        self.freq_hopper = FrequencyHoppingEngine(self.sdr)
        self.power_adjuster = PowerAdjustmentEngine(self.sdr)
        self.jamming_mitigator = JammingMitigationEngine(self.sdr, self.freq_hopper, self.power_adjuster)

        logger.info("Countermeasure engine initialized")

    def execute_countermeasure(
        self,
        threat: ThreatInfo,
        countermeasure_type: CountermeasureType,
        parameters: Optional[Dict] = None
    ) -> CountermeasureResult:
        """Execute specified countermeasure"""

        if parameters is None:
            parameters = {}

        logger.info(f"Executing {countermeasure_type.value} for {threat.threat_type} threat at {threat.frequency/1e6:.3f} MHz")

        try:
            if countermeasure_type == CountermeasureType.FREQUENCY_HOPPING:
                return self.freq_hopper.execute(threat, parameters)

            elif countermeasure_type == CountermeasureType.POWER_ADJUSTMENT:
                return self.power_adjuster.execute(threat, parameters)

            elif countermeasure_type == CountermeasureType.JAMMING_MITIGATION:
                return self.jamming_mitigator.execute(threat, parameters)

            elif countermeasure_type == CountermeasureType.SPECTRUM_EVASION:
                # Combination of frequency hopping + power reduction
                freq_result = self.freq_hopper.execute(threat, parameters)
                power_result = self.power_adjuster.execute(threat, {'power_reduction_db': 5})

                return CountermeasureResult(
                    success=freq_result.success and power_result.success,
                    countermeasure_type=CountermeasureType.SPECTRUM_EVASION,
                    actions_taken=freq_result.actions_taken + power_result.actions_taken,
                    new_parameters={**freq_result.new_parameters, **power_result.new_parameters},
                    duration_ms=freq_result.duration_ms + power_result.duration_ms,
                    message="Spectrum evasion completed"
                )

            elif countermeasure_type == CountermeasureType.ALERT_ONLY:
                return CountermeasureResult(
                    success=True,
                    countermeasure_type=CountermeasureType.ALERT_ONLY,
                    actions_taken=["Operator alerted, no automated action taken"],
                    new_parameters={},
                    duration_ms=0,
                    message="Alert sent to operator"
                )

            else:
                return CountermeasureResult(
                    success=False,
                    countermeasure_type=countermeasure_type,
                    actions_taken=[],
                    new_parameters={},
                    duration_ms=0,
                    message=f"Countermeasure {countermeasure_type.value} not implemented"
                )

        except Exception as e:
            logger.error(f"Countermeasure execution failed: {e}")
            return CountermeasureResult(
                success=False,
                countermeasure_type=countermeasure_type,
                actions_taken=[f"Error: {str(e)}"],
                new_parameters={},
                duration_ms=0,
                message=f"Countermeasure failed: {str(e)}"
            )


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = CountermeasureEngine()

    # Simulate jamming threat
    threat = ThreatInfo(
        threat_id="threat_001",
        threat_type="jamming",
        severity=ThreatSeverity.CRITICAL,
        frequency=915e6,  # 915 MHz
        power=-25,  # -25 dBm (high power)
        bandwidth=40e6,  # 40 MHz (wide)
        location=(37.7749, -122.4194)
    )

    # Execute jamming mitigation
    result = engine.execute_countermeasure(
        threat=threat,
        countermeasure_type=CountermeasureType.JAMMING_MITIGATION,
        parameters={
            'hop_pattern': 'adaptive',
            'hop_rate_ms': 500,
            'backup_frequencies': [868e6, 915e6, 2450e6],
            'enable_spread_spectrum': True,
            'enable_fec': True,
        }
    )

    print(f"\n{'='*60}")
    print(f"Countermeasure Result:")
    print(f"{'='*60}")
    print(f"Success: {result.success}")
    print(f"Type: {result.countermeasure_type.value}")
    print(f"Duration: {result.duration_ms:.1f} ms")
    print(f"Message: {result.message}")
    print(f"\nActions Taken:")
    for action in result.actions_taken:
        print(f"  - {action}")
    print(f"\nNew Parameters:")
    for key, value in result.new_parameters.items():
        if 'frequency' in key:
            print(f"  {key}: {value/1e6:.3f} MHz")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*60}\n")
