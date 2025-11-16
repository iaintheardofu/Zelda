"""
ZELDA Advanced SIGINT Module

World-class signal intelligence processing with:
- Advanced detection algorithms (cyclostationary, energy, blind)
- Modulation classification (50+ types)
- Signal characterization and fingerprinting
- Cognitive radio and interference mitigation
"""

__version__ = "2.0.0"
__author__ = "ZELDA Development Team"

from .detection import (
    CyclostationaryDetector,
    EnergyDetector,
    BlindDetector,
    MultiAlgorithmFusion,
)

from .modulation import (
    ModulationClassifier,
    SignalCharacterizer,
    EmitterFingerprint,
)

from .cognitive_radio import (
    CognitiveEngine,
    InterferenceCanceller,
    SpectrumManager,
    AdaptiveFilter,
)

__all__ = [
    'CyclostationaryDetector',
    'EnergyDetector',
    'BlindDetector',
    'MultiAlgorithmFusion',
    'ModulationClassifier',
    'SignalCharacterizer',
    'EmitterFingerprint',
    'CognitiveEngine',
    'InterferenceCanceller',
    'SpectrumManager',
    'AdaptiveFilter',
]
