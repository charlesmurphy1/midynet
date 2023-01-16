from typing import Dict, Any
class PredictionHeuristicsMethod:
    def __init__(self, graph_metrics: Dict[str, Any], dynamics_metrics: Dict[str, Any]):
        self.
    def extract_structural_features(self, graph):
        raise NotImplementedError()
    def fit(self, graph, timeseries):
        x = self.extract_structural_features(graph)
        y = self.extract_dynamic_feature(timeseries)