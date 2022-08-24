import midynet
import numpy as np
import matplotlib.pyplot as plt


def main():
    config = midynet.config.ExperimentConfig.reconstruction(
        "test", "sis", "erdosrenyi", metrics="heuristics"
    )
    config.graph_prior.set_value("size", 100)
    config.graph_prior.set_value("edge_count", 250)
    config.data_model.set_value("num_steps", 1000)
    config.data_model.set_value("infection_prob", 0.5)
    config.metrics.heuristics.set_value("method", "granger_causality")
    graph_model = midynet.config.RandomGraphFactory.build(config.graph_prior)
    data_model = midynet.config.DataModelFactory.build(config.data_model)
    data_model.set_graph_prior(graph_model)

    data_model.sample()
    timeseries = np.array(data_model.get_past_states()).T
    heuristics = midynet.metrics.heuristics.get_heuristics_reconstructor(
        config.metrics.heuristics
    )
    heuristics.fit(timeseries)
    heuristics.compare(graph_model.get_state(), collectors=["roc"])
    print(heuristics.__results__["roc"]["auc"])


if __name__ == "__main__":
    main()
