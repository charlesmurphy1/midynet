import midynet
import numpy as np


def main():
    N, E, B = 5, 3, 2
    test_model = midynet.random_graph.PlantedPartitionModel(N, E, B, assortativity=0.8)
    graph_model = midynet.random_graph.StochasticBlockModelFamily(
        N,
        E,
        block_prior_type="hyper",
        block_proposer_type="mixed",
        label_graph_prior_type="nested",
    )
    graph_model.set_state(test_model.get_state())
    config = midynet.config.Config(
        method="meanfield",
        num_sweeps=1000,
        burn_per_vertex=10,
        equilibrate_mode_cluster=True,
    )
    joint = graph_model.get_log_joint()
    mf = midynet.metrics.util.get_graph_log_evidence_meanfield(graph_model, config)
    annealed = midynet.metrics.util.get_graph_log_evidence_annealed(graph_model, config)
    exact = midynet.metrics.util.get_graph_log_evidence_exact(graph_model, config)


if __name__ == "__main__":
    main()
