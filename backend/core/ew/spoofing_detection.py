"""
ZELDA Defensive EW Suite - Spoofing Detection Module

Detects GPS, cellular, and WiFi spoofing attacks.
All capabilities are DEFENSIVE - detection and analysis only, no transmission.

Legal Use: Security monitoring, threat detection, authorized testing
"""

import numpy as np
from scipy import signal, stats
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SpoofingType(Enum):
    """Types of spoofing attacks that can be detected"""
    NONE = "none"
    GPS_MEACONING = "gps_meaconing"  # Replay attack
    GPS_SIMULATION = "gps_simulation"  # Fake GPS signals
    CELLULAR_IMSI_CATCHER = "imsi_catcher"  # Fake base station
    CELLULAR_FEMTOCELL = "rogue_femtocell"  # Unauthorized small cell
    WIFI_EVIL_TWIN = "wifi_evil_twin"  # Fake access point
    WIFI_ROGUE_AP = "wifi_rogue_ap"  # Unauthorized AP
    BLUETOOTH_SPOOF = "bluetooth_spoof"  # Fake Bluetooth device
    UNKNOWN = "unknown"


@dataclass
class SpoofingDetection:
    """Results from spoofing detection analysis"""
    is_spoofed: bool
    spoofing_type: SpoofingType
    confidence: float  # 0.0 to 1.0
    threat_level: str  # "low", "medium", "high", "critical"
    detection_time: datetime
    indicators: List[str]  # List of anomaly indicators
    characteristics: Dict  # Detailed characteristics
    recommendations: List[str]  # Mitigation recommendations


class GPSSpoofingDetector:
    """
    Detect GPS spoofing attacks through signal analysis.

    Detection Methods:
    - Multiple inconsistent GPS signals
    - Power level anomalies
    - Clock offset inconsistencies
    - Doppler shift anomalies
    - C/N0 (carrier-to-noise) analysis
    """

    def __init__(self, sample_rate: float = 10e6):
        self.sample_rate = sample_rate
        self.gps_l1_freq = 1575.42e6  # GPS L1 frequency
        self.expected_power_range_db = (-160, -140)  # Typical GPS power at antenna

        # Historical tracking
        self.signal_history = []
        self.power_history = []
        self.max_history = 50

        logger.info("GPSSpoofingDetector initialized")

    def detect(self, iq_signal: np.ndarray, metadata: Optional[Dict] = None) -> SpoofingDetection:
        """
        Analyze GPS signal for spoofing indicators.

        Args:
            iq_signal: Complex I/Q samples
            metadata: Optional metadata (power levels, satellite info, etc.)

        Returns:
            SpoofingDetection with analysis results
        """
        indicators = []
        characteristics = {}
        confidence = 0.0
        spoofing_type = SpoofingType.NONE

        # Analysis 1: Power level anomalies
        power_db = self._estimate_signal_power(iq_signal)
        characteristics['power_db'] = power_db

        if power_db > self.expected_power_range_db[1]:
            indicators.append("Abnormally high GPS signal power")
            confidence += 0.3
            spoofing_type = SpoofingType.GPS_SIMULATION

        # Analysis 2: Multiple signal detection
        num_signals = self._count_gps_signals(iq_signal)
        characteristics['num_signals_detected'] = num_signals

        if num_signals > 12:  # More than typical visible satellites
            indicators.append(f"Excessive GPS signals detected ({num_signals} > 12)")
            confidence += 0.25

        # Analysis 3: Signal correlation (meaconing detection)
        correlation_score = self._detect_signal_correlation(iq_signal)
        characteristics['correlation_score'] = correlation_score

        if correlation_score > 0.9:  # High correlation indicates replay
            indicators.append("High correlation between signals (replay attack suspected)")
            confidence += 0.3
            spoofing_type = SpoofingType.GPS_MEACONING

        # Analysis 4: Timing inconsistencies
        if metadata and 'timestamps' in metadata:
            timing_anomaly = self._check_timing_consistency(metadata['timestamps'])
            if timing_anomaly:
                indicators.append("GPS timing inconsistencies detected")
                confidence += 0.2

        # Analysis 5: Sudden position jumps
        if metadata and 'position' in metadata:
            position_jump = self._detect_position_jump(metadata['position'])
            characteristics['position_jump_meters'] = position_jump

            if position_jump > 100:  # >100m sudden jump
                indicators.append(f"Sudden position jump: {position_jump:.0f}m")
                confidence += 0.25

        # Analysis 6: C/N0 consistency check
        cn0_values = self._estimate_cn0(iq_signal, num_signals)
        characteristics['cn0_db_hz'] = np.mean(cn0_values) if cn0_values else None

        if cn0_values and np.std(cn0_values) < 1.0:  # Too uniform
            indicators.append("Suspiciously uniform C/N0 across satellites")
            confidence += 0.15

        # Determine if spoofed
        is_spoofed = confidence >= 0.5

        # Determine threat level
        if confidence < 0.3:
            threat_level = "low"
        elif confidence < 0.6:
            threat_level = "medium"
        elif confidence < 0.8:
            threat_level = "high"
        else:
            threat_level = "critical"

        # Generate recommendations
        recommendations = self._generate_recommendations(spoofing_type, indicators)

        return SpoofingDetection(
            is_spoofed=is_spoofed,
            spoofing_type=spoofing_type if is_spoofed else SpoofingType.NONE,
            confidence=min(confidence, 1.0),
            threat_level=threat_level,
            detection_time=datetime.now(),
            indicators=indicators,
            characteristics=characteristics,
            recommendations=recommendations
        )

    def _estimate_signal_power(self, iq_signal: np.ndarray) -> float:
        """Estimate signal power in dBm"""
        power_linear = np.mean(np.abs(iq_signal) ** 2)
        power_db = 10 * np.log10(power_linear + 1e-12)
        return power_db

    def _count_gps_signals(self, iq_signal: np.ndarray) -> int:
        """Count number of distinct GPS signals using peak detection"""
        # Compute power spectral density
        freqs, psd = signal.welch(iq_signal, fs=self.sample_rate, nperseg=1024)

        # Find peaks
        peaks, _ = signal.find_peaks(psd, height=np.median(psd) * 3, distance=10)

        return len(peaks)

    def _detect_signal_correlation(self, iq_signal: np.ndarray) -> float:
        """
        Detect if multiple GPS signals are highly correlated (replay attack).

        Returns correlation score (0.0 to 1.0)
        """
        # Split signal into chunks
        chunk_size = len(iq_signal) // 4
        chunks = [
            iq_signal[i*chunk_size:(i+1)*chunk_size]
            for i in range(4)
        ]

        # Calculate cross-correlations
        correlations = []
        for i in range(len(chunks)):
            for j in range(i+1, len(chunks)):
                corr = np.corrcoef(np.abs(chunks[i]), np.abs(chunks[j]))[0, 1]
                correlations.append(abs(corr))

        return np.mean(correlations) if correlations else 0.0

    def _check_timing_consistency(self, timestamps: List[float]) -> bool:
        """Check for timing anomalies"""
        if len(timestamps) < 2:
            return False

        # Calculate time differences
        diffs = np.diff(timestamps)

        # Check for irregular spacing (should be ~1 second for GPS)
        expected_interval = 1.0
        tolerance = 0.1

        anomalies = np.sum(np.abs(diffs - expected_interval) > tolerance)
        return anomalies > len(diffs) * 0.2  # >20% anomalous

    def _detect_position_jump(self, position: Tuple[float, float, float]) -> float:
        """
        Detect sudden position jumps.

        Args:
            position: (lat, lon, alt) tuple

        Returns:
            Jump distance in meters
        """
        if not hasattr(self, 'last_position') or self.last_position is None:
            self.last_position = position
            return 0.0

        # Calculate distance (simplified)
        lat1, lon1, alt1 = self.last_position
        lat2, lon2, alt2 = position

        # Haversine formula (approximate)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = 6371000 * c  # Earth radius in meters

        self.last_position = position
        return distance

    def _estimate_cn0(self, iq_signal: np.ndarray, num_signals: int) -> List[float]:
        """Estimate C/N0 for each detected GPS signal"""
        # Simplified C/N0 estimation
        cn0_values = []

        for _ in range(num_signals):
            # Estimate signal and noise power
            signal_power = np.var(np.abs(iq_signal))
            noise_power = np.percentile(np.abs(iq_signal)**2, 10)

            if noise_power > 0:
                cn0_db = 10 * np.log10(signal_power / noise_power)
                cn0_values.append(cn0_db)

        return cn0_values

    def _generate_recommendations(self, spoofing_type: SpoofingType, indicators: List[str]) -> List[str]:
        """Generate mitigation recommendations"""
        recommendations = []

        if spoofing_type == SpoofingType.GPS_MEACONING:
            recommendations.append("Enable anti-replay protection (if available)")
            recommendations.append("Cross-check with inertial navigation system")
            recommendations.append("Monitor for signal power anomalies")

        elif spoofing_type == SpoofingType.GPS_SIMULATION:
            recommendations.append("Verify GPS authenticity with multi-frequency receiver")
            recommendations.append("Use GPS authentication (if available)")
            recommendations.append("Cross-reference with alternative PNT sources")

        if len(indicators) > 0:
            recommendations.append("Log all GPS data for forensic analysis")
            recommendations.append("Alert security operations center")

        return recommendations


class CellularSpoofingDetector:
    """
    Detect cellular network spoofing (IMSI catchers, rogue base stations).

    Detection Methods:
    - Signal strength anomalies
    - Cell ID inconsistencies
    - LAC/TAC changes
    - Downgrade attacks (4G → 2G)
    - Encryption status changes
    """

    def __init__(self):
        # Known cell tower database (would be populated in production)
        self.known_cells = {}
        self.cell_history = []

        logger.info("CellularSpoofingDetector initialized")

    def detect(self, cell_info: Dict) -> SpoofingDetection:
        """
        Analyze cellular connection for spoofing indicators.

        Args:
            cell_info: Dictionary with cell tower information
                {
                    'cell_id': int,
                    'lac': int (Location Area Code),
                    'mcc': int (Mobile Country Code),
                    'mnc': int (Mobile Network Code),
                    'signal_strength_dbm': float,
                    'network_type': str ('2G', '3G', '4G', '5G'),
                    'encryption': bool
                }

        Returns:
            SpoofingDetection with analysis results
        """
        indicators = []
        characteristics = {}
        confidence = 0.0
        spoofing_type = SpoofingType.NONE

        # Analysis 1: Unknown cell tower
        cell_key = (cell_info['mcc'], cell_info['mnc'], cell_info['cell_id'])
        is_known = cell_key in self.known_cells
        characteristics['known_cell'] = is_known

        if not is_known and len(self.known_cells) > 0:
            indicators.append(f"Unknown cell tower detected: Cell ID {cell_info['cell_id']}")
            confidence += 0.4
            spoofing_type = SpoofingType.CELLULAR_IMSI_CATCHER

        # Analysis 2: Signal strength anomaly
        signal_strength = cell_info.get('signal_strength_dbm', -100)
        characteristics['signal_strength_dbm'] = signal_strength

        if signal_strength > -50:  # Unusually strong (very close)
            indicators.append(f"Abnormally strong signal: {signal_strength} dBm")
            confidence += 0.3

        # Analysis 3: Network downgrade attack
        if hasattr(self, 'last_network_type'):
            current_type = cell_info.get('network_type', '4G')
            if self._is_downgrade(self.last_network_type, current_type):
                indicators.append(f"Network downgrade detected: {self.last_network_type} → {current_type}")
                confidence += 0.35
                spoofing_type = SpoofingType.CELLULAR_IMSI_CATCHER

        self.last_network_type = cell_info.get('network_type', '4G')

        # Analysis 4: Encryption disabled
        if not cell_info.get('encryption', True):
            indicators.append("Cellular encryption disabled (A5/0 or no encryption)")
            confidence += 0.4
            spoofing_type = SpoofingType.CELLULAR_IMSI_CATCHER

        # Analysis 5: LAC/TAC changes without movement
        if 'lac' in cell_info:
            if hasattr(self, 'last_lac') and cell_info['lac'] != self.last_lac:
                # LAC change without significant movement suggests rogue cell
                indicators.append(f"Location Area Code changed: {self.last_lac} → {cell_info['lac']}")
                confidence += 0.25

            self.last_lac = cell_info['lac']

        # Analysis 6: Multiple cells with same ID (femtocell attack)
        if self._detect_duplicate_cell_ids(cell_info):
            indicators.append("Multiple cells with identical ID detected")
            confidence += 0.3
            spoofing_type = SpoofingType.CELLULAR_FEMTOCELL

        # Store in history
        self.cell_history.append(cell_info)
        if len(self.cell_history) > 100:
            self.cell_history.pop(0)

        # Determine if spoofed
        is_spoofed = confidence >= 0.5

        # Determine threat level
        if confidence < 0.3:
            threat_level = "low"
        elif confidence < 0.6:
            threat_level = "medium"
        elif confidence < 0.8:
            threat_level = "high"
        else:
            threat_level = "critical"

        # Generate recommendations
        recommendations = self._generate_recommendations(spoofing_type, indicators)

        return SpoofingDetection(
            is_spoofed=is_spoofed,
            spoofing_type=spoofing_type if is_spoofed else SpoofingType.NONE,
            confidence=min(confidence, 1.0),
            threat_level=threat_level,
            detection_time=datetime.now(),
            indicators=indicators,
            characteristics=characteristics,
            recommendations=recommendations
        )

    def _is_downgrade(self, old_type: str, new_type: str) -> bool:
        """Check if network type represents a downgrade"""
        hierarchy = {'5G': 4, '4G': 3, '3G': 2, '2G': 1}
        old_level = hierarchy.get(old_type, 0)
        new_level = hierarchy.get(new_type, 0)
        return new_level < old_level

    def _detect_duplicate_cell_ids(self, cell_info: Dict) -> bool:
        """Detect multiple cells with same ID at different locations"""
        current_cell_id = cell_info['cell_id']

        # Check recent history for same cell ID at different signal strengths
        matching_cells = [
            c for c in self.cell_history[-20:]
            if c['cell_id'] == current_cell_id
        ]

        if len(matching_cells) < 2:
            return False

        # Check if signal strengths vary significantly
        signal_strengths = [c.get('signal_strength_dbm', -100) for c in matching_cells]
        variance = np.var(signal_strengths)

        return variance > 100  # High variance suggests different physical locations

    def _generate_recommendations(self, spoofing_type: SpoofingType, indicators: List[str]) -> List[str]:
        """Generate mitigation recommendations"""
        recommendations = []

        if spoofing_type == SpoofingType.CELLULAR_IMSI_CATCHER:
            recommendations.append("Disable 2G/3G (force 4G/5G only if supported)")
            recommendations.append("Avoid sensitive communications until verified")
            recommendations.append("Use encrypted messaging (Signal, WhatsApp)")
            recommendations.append("Report to carrier security team")

        elif spoofing_type == SpoofingType.CELLULAR_FEMTOCELL:
            recommendations.append("Verify femtocell authorization with carrier")
            recommendations.append("Check for unauthorized devices in area")

        if len(indicators) > 0:
            recommendations.append("Enable VPN for all data traffic")
            recommendations.append("Monitor for unusual account activity")

        return recommendations


class WiFiSpoofingDetector:
    """
    Detect WiFi spoofing attacks (Evil Twin, Rogue APs).

    Detection Methods:
    - SSID duplicates with different BSSIDs
    - Signal strength anomalies
    - Encryption downgrade
    - Unusual channel usage
    - Vendor OUI analysis
    """

    def __init__(self):
        self.known_aps = {}  # Known legitimate APs
        self.ap_history = []

        logger.info("WiFiSpoofingDetector initialized")

    def detect(self, ap_info: Dict) -> SpoofingDetection:
        """
        Analyze WiFi access point for spoofing indicators.

        Args:
            ap_info: Dictionary with AP information
                {
                    'ssid': str,
                    'bssid': str (MAC address),
                    'signal_strength_dbm': float,
                    'channel': int,
                    'encryption': str ('Open', 'WEP', 'WPA', 'WPA2', 'WPA3'),
                    'vendor': str (optional)
                }

        Returns:
            SpoofingDetection with analysis results
        """
        indicators = []
        characteristics = {}
        confidence = 0.0
        spoofing_type = SpoofingType.NONE

        ssid = ap_info['ssid']
        bssid = ap_info['bssid']

        # Analysis 1: Evil Twin detection (same SSID, different BSSID)
        if ssid in self.known_aps:
            known_bssid = self.known_aps[ssid]['bssid']
            if bssid != known_bssid:
                indicators.append(f"Duplicate SSID with different BSSID: {bssid} vs {known_bssid}")
                confidence += 0.5
                spoofing_type = SpoofingType.WIFI_EVIL_TWIN

        # Analysis 2: Encryption downgrade
        if ssid in self.known_aps:
            known_encryption = self.known_aps[ssid]['encryption']
            current_encryption = ap_info['encryption']

            if self._is_encryption_downgrade(known_encryption, current_encryption):
                indicators.append(f"Encryption downgrade: {known_encryption} → {current_encryption}")
                confidence += 0.4

        # Analysis 3: Abnormal signal strength
        signal_strength = ap_info.get('signal_strength_dbm', -100)
        characteristics['signal_strength_dbm'] = signal_strength

        if signal_strength > -30:  # Extremely strong
            indicators.append(f"Abnormally strong WiFi signal: {signal_strength} dBm")
            confidence += 0.2

        # Analysis 4: Unusual channel
        channel = ap_info.get('channel', 6)
        if channel > 14 and channel not in [36, 40, 44, 48, 149, 153, 157, 161]:
            indicators.append(f"Non-standard WiFi channel: {channel}")
            confidence += 0.15

        # Analysis 5: Open network with common SSID
        if ap_info['encryption'] == 'Open' and self._is_common_ssid(ssid):
            indicators.append(f"Open network with common SSID: {ssid}")
            confidence += 0.3
            spoofing_type = SpoofingType.WIFI_ROGUE_AP

        # Store AP information
        if ssid not in self.known_aps:
            self.known_aps[ssid] = ap_info

        self.ap_history.append(ap_info)
        if len(self.ap_history) > 100:
            self.ap_history.pop(0)

        # Determine if spoofed
        is_spoofed = confidence >= 0.4  # Lower threshold for WiFi

        # Determine threat level
        if confidence < 0.3:
            threat_level = "low"
        elif confidence < 0.5:
            threat_level = "medium"
        elif confidence < 0.7:
            threat_level = "high"
        else:
            threat_level = "critical"

        # Generate recommendations
        recommendations = self._generate_recommendations(spoofing_type, indicators)

        return SpoofingDetection(
            is_spoofed=is_spoofed,
            spoofing_type=spoofing_type if is_spoofed else SpoofingType.NONE,
            confidence=min(confidence, 1.0),
            threat_level=threat_level,
            detection_time=datetime.now(),
            indicators=indicators,
            characteristics=characteristics,
            recommendations=recommendations
        )

    def _is_encryption_downgrade(self, old_enc: str, new_enc: str) -> bool:
        """Check if encryption represents a downgrade"""
        hierarchy = {'WPA3': 4, 'WPA2': 3, 'WPA': 2, 'WEP': 1, 'Open': 0}
        old_level = hierarchy.get(old_enc, 0)
        new_level = hierarchy.get(new_enc, 0)
        return new_level < old_level

    def _is_common_ssid(self, ssid: str) -> bool:
        """Check if SSID is commonly targeted in evil twin attacks"""
        common_ssids = [
            'Free WiFi', 'Free Public WiFi', 'Airport WiFi',
            'Starbucks', 'McDonalds', 'Hotel WiFi',
            'Guest', 'Public', 'attwifi'
        ]
        return any(common.lower() in ssid.lower() for common in common_ssids)

    def _generate_recommendations(self, spoofing_type: SpoofingType, indicators: List[str]) -> List[str]:
        """Generate mitigation recommendations"""
        recommendations = []

        if spoofing_type == SpoofingType.WIFI_EVIL_TWIN:
            recommendations.append("DO NOT connect to this network")
            recommendations.append("Forget this network from device")
            recommendations.append("Verify AP MAC address with network administrator")
            recommendations.append("Use VPN if connection is necessary")

        elif spoofing_type == SpoofingType.WIFI_ROGUE_AP:
            recommendations.append("Avoid connecting to open networks")
            recommendations.append("Use cellular data instead")
            recommendations.append("If must connect, use VPN immediately")

        if len(indicators) > 0:
            recommendations.append("Enable 'Require WPA3' in WiFi settings")
            recommendations.append("Disable auto-connect for untrusted networks")

        return recommendations


# Integrated spoofing detection system
class IntegratedSpoofingDetector:
    """
    Unified spoofing detection across GPS, Cellular, and WiFi.
    """

    def __init__(self):
        self.gps_detector = GPSSpoofingDetector()
        self.cellular_detector = CellularSpoofingDetector()
        self.wifi_detector = WiFiSpoofingDetector()

        logger.info("IntegratedSpoofingDetector initialized")

    def detect_all(
        self,
        gps_signal: Optional[np.ndarray] = None,
        cell_info: Optional[Dict] = None,
        wifi_aps: Optional[List[Dict]] = None
    ) -> Dict[str, SpoofingDetection]:
        """
        Run all spoofing detection systems.

        Returns:
            Dictionary with results for each system
        """
        results = {}

        if gps_signal is not None:
            results['gps'] = self.gps_detector.detect(gps_signal)

        if cell_info is not None:
            results['cellular'] = self.cellular_detector.detect(cell_info)

        if wifi_aps is not None:
            wifi_results = []
            for ap in wifi_aps:
                wifi_results.append(self.wifi_detector.detect(ap))
            results['wifi'] = wifi_results

        return results

    def generate_summary_report(self, results: Dict) -> str:
        """Generate comprehensive spoofing detection report"""
        report = []
        report.append("=" * 70)
        report.append("ZELDA INTEGRATED SPOOFING DETECTION REPORT")
        report.append("=" * 70)
        report.append(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        total_threats = 0

        # GPS Results
        if 'gps' in results:
            gps = results['gps']
            report.append("GPS SPOOFING ANALYSIS:")
            report.append(f"  Status: {'⚠️  SPOOFED' if gps.is_spoofed else '✓  Clean'}")
            if gps.is_spoofed:
                report.append(f"  Type: {gps.spoofing_type.value}")
                report.append(f"  Confidence: {gps.confidence*100:.1f}%")
                report.append(f"  Threat Level: {gps.threat_level.upper()}")
                total_threats += 1
            report.append("")

        # Cellular Results
        if 'cellular' in results:
            cell = results['cellular']
            report.append("CELLULAR SPOOFING ANALYSIS:")
            report.append(f"  Status: {'⚠️  SPOOFED' if cell.is_spoofed else '✓  Clean'}")
            if cell.is_spoofed:
                report.append(f"  Type: {cell.spoofing_type.value}")
                report.append(f"  Confidence: {cell.confidence*100:.1f}%")
                report.append(f"  Threat Level: {cell.threat_level.upper()}")
                total_threats += 1
            report.append("")

        # WiFi Results
        if 'wifi' in results:
            wifi_list = results['wifi']
            report.append(f"WIFI SPOOFING ANALYSIS ({len(wifi_list)} networks scanned):")
            spoofed_count = sum(1 for w in wifi_list if w.is_spoofed)
            report.append(f"  Status: {'⚠️  {spoofed_count} SUSPICIOUS' if spoofed_count > 0 else '✓  All Clean'}")

            for i, wifi in enumerate(wifi_list):
                if wifi.is_spoofed:
                    report.append(f"  Network {i+1}: {wifi.spoofing_type.value} ({wifi.confidence*100:.0f}% confidence)")
                    total_threats += 1
            report.append("")

        # Summary
        report.append("SUMMARY:")
        report.append(f"  Total Threats Detected: {total_threats}")

        if total_threats > 0:
            report.append("")
            report.append("⚠️  RECOMMENDED ACTIONS:")
            # Collect all recommendations
            all_recommendations = set()
            for key in results:
                if isinstance(results[key], list):
                    for r in results[key]:
                        all_recommendations.update(r.recommendations)
                else:
                    all_recommendations.update(results[key].recommendations)

            for i, rec in enumerate(sorted(all_recommendations), 1):
                report.append(f"  {i}. {rec}")

        report.append("=" * 70)
        report.append("All detections are DEFENSIVE analysis only (no transmission)")
        report.append("=" * 70)

        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    print("ZELDA Defensive EW - Spoofing Detection Module")
    print("=" * 60)
    print("Legal Use: Detection and analysis only (no transmission)")
    print("=" * 60)

    # Create integrated detector
    detector = IntegratedSpoofingDetector()

    # Test GPS spoofing detection
    print("\n--- GPS Spoofing Test ---")
    # Simulate GPS signal (simplified)
    t = np.linspace(0, 1024/10e6, 1024)
    gps_signal = 2 * np.exp(1j * 2 * np.pi * 1575.42e6 * t)  # Strong GPS signal (suspicious)

    # Test cellular spoofing detection
    print("\n--- Cellular Spoofing Test ---")
    cell_info = {
        'cell_id': 12345,
        'lac': 100,
        'mcc': 310,  # USA
        'mnc': 260,  # T-Mobile
        'signal_strength_dbm': -45,  # Very strong (suspicious)
        'network_type': '2G',  # Downgrade
        'encryption': False  # No encryption (very suspicious)
    }

    # Test WiFi spoofing detection
    print("\n--- WiFi Spoofing Test ---")
    wifi_aps = [
        {
            'ssid': 'Starbucks WiFi',
            'bssid': '00:11:22:33:44:55',
            'signal_strength_dbm': -35,  # Very strong
            'channel': 6,
            'encryption': 'Open'  # Suspicious for Starbucks
        },
        {
            'ssid': 'Home Network',
            'bssid': 'AA:BB:CC:DD:EE:FF',
            'signal_strength_dbm': -60,
            'channel': 36,
            'encryption': 'WPA2'
        }
    ]

    # Run all detections
    results = detector.detect_all(
        gps_signal=gps_signal,
        cell_info=cell_info,
        wifi_aps=wifi_aps
    )

    # Print comprehensive report
    print(detector.generate_summary_report(results))

    print("\n✓ Spoofing detection module operational")
    print("✓ All capabilities are DEFENSIVE (detection/analysis only)")
