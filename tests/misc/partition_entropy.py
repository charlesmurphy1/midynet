import midynet
import numpy as np


def main():
    N, E, B = 100, 250, 5
    test_model = midynet.random_graph.PlantedPartitionModel(N, E, B)
    prior_model = midynet.random_graph.StochasticBlockModelFamily(
        N, E, block_hyperprior=True, block_proposer_type="mixed"
    )
    prior_model.set_state(test_model.get_state())
    mf = midynet.metrics.util.get_posterior_entropy_partition_meanfield(
        prior_model, num_sweeps=1000, burn_per_vertex=10
    )
    print(mf)


if __name__ == "__main__":
    main()
