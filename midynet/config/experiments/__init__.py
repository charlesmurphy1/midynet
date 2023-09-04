from .duality.duality_timestep import DualityTimestepScriptConfig
from .duality.mi_vs_heuristics import (
    PredHeuristicsScriptConfig,
    ReconHeuristicsScriptConfig,
)
from .duality.duality_coupling import (
    DualityCouplingWithRealGraphsScriptConfig,
    DualityCouplingSyntheticGraphsScriptConfig,
)

from .recon.error_heuristics import ErrorHeuristicsScriptConfig
from .recon.infogain_syn import InfoGainSyntheticGraphsScriptConfig

__all_configs__ = {
    "duality.duality-timestep": DualityTimestepScriptConfig,
    "duality.pred_heuristics": PredHeuristicsScriptConfig,
    "duality.recon_heuristics": ReconHeuristicsScriptConfig,
    "duality.duality-coupling-with-rn": DualityCouplingWithRealGraphsScriptConfig,
    "duality.duality-coupling-with-syn": DualityCouplingSyntheticGraphsScriptConfig,
    "recon.error-heuristics": ErrorHeuristicsScriptConfig,
    "recon.infogain-syn": InfoGainSyntheticGraphsScriptConfig,
}
__all__ = (
    "DualityTimestepScriptConfig",
    "PredHeuristicsScriptConfig",
    "ReconHeuristicsScriptConfig",
    "DualityCouplingWithRealGraphsScriptConfig",
    "DualityCouplingSyntheticGraphsScriptConfig",
    "ErrorHeuristicsScriptConfig",
    "InfoGainSyntheticGraphsScriptConfig",
)
