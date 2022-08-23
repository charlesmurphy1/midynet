import midynet
import numpy as np


def main():
    N, E, B = 3, 3, 2
    test_model = midynet.random_graph.PlantedPartitionModel(N, E, B)
    graph_model = midynet.random_graph.StochasticBlockModelFamily(
        N, E, block_prior_type="hyper", block_proposer_type="mixed"
    )
    graph_model.set_state(test_model.get_state())
    mf = midynet.metrics.util.get_posterior_entropy_partition_meanfield(
        graph_model, num_sweeps=1000, burn_per_vertex=10
    )

    exact = midynet.metrics.util.get_posterior_entropy_partition_exact(graph_model)
    print(mf, exact)


if __name__ == "__main__":
    main()
