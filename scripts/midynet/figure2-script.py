import midynet

num_procs = 32
config = midynet.config.Experiment(
    "figure-2-exact",
    ["ising", "sis", "cowan"],
    "ser",
    metrics=["mutualinfo"],
    path=".",
    # num_procs=32,
)
config.metrics.mutualinfo.set_value("num_samples", 500 // num_procs * num_procs)
config.metrics.mutualinfo.set_value("num_sweeps", 100)
config.metrics.mutualinfo.set_value("burn_per_vertex", 100)
config.metrics.mutualinfo.set_value("method", "exact")

config.set_value("dynamics.ising.num_steps", [10, 25, 50, 100, 250, 500, 1000])
config.set_value("dynamics.sis.num_steps", [10, 25, 50, 100, 250, 500, 1000])
config.set_value("dynamics.cowan.num_steps", [10, 25, 50, 100, 250, 500, 1000])

config.set_value("dynamics.ising.coupling", [10, 25, 50, 100, 250, 500, 1000])
config.set_value("dynamics.sis.infection_prob", [10, 25, 50, 100, 250, 500, 1000])
config.set_value("dynamics.cowan.nu", [10, 25, 50, 100, 250, 500, 1000])


num_procs = 32
dynamics = [dynamics_name]
fig2_config = {
    # "dynamics/normalize": True,
    "metrics/num_samples": 96,
    "metrics/num_sweeps": 100,
    "metrics/num_steps": 100,
    "metrics/kmax": 20,
    "metrics/adaptive": False,
    "metrics/max_flips": 5,
    "metrics/init_flips": 1,
    "metrics/error_type": "confidence",
    "num_procs": num_procs,
}

N = 5
config["graph/num_vertices"] = [N]
config["graph/num_edges"] = tuple(range(1, int(N * (N - 1) / 2)))
config["metrics/num_steps"] = (10, 25, 50, 100, 250, 500, 1000)
config["metrics/method"] = ["exact"]
config["metrics/num_samples"] = 500 // config["num_procs"] * config["num_procs"]


launching(config, "24:00:00")
