import tempfile
import os
import dotenv

from math import ceil
from midynet.config import ExperimentConfig, GraphConfig, DataModelConfig
from util import format_sequence

dotenv.load_dotenv()
PATH_TO_DATA = os.getenv("PATH_TO_DATA", "../../data")
MAXNUMJOBS = int(os.getenv("MAXNUMJOBS", 4))
MAXMEMORY = int(os.getenv("MAXMEMORY", 0))
ACCOUNT = os.getenv("ACCOUNT", None)


prior_dict = {
    "erdosrenyi": GraphConfig.erdosrenyi(
        size=100, edge_count=250, loopy=False, multigraph=False
    )
}

model_dict = {
    "glauber": DataModelConfig.glauber(
        length=100, coupling=format_sequence((0, 0.8, 25))
    )
}


class ErrorHeuristicsScriptConfig(ExperimentConfig):
    @classmethod
    def default(
        cls,
        save_path=None,
        n_workers=MAXNUMJOBS,
        n_samples_per_worker=1,
        n_async_jobs=1,
        time="24:00:00",
        mem=MAXMEMORY,
        seed=None,
    ):
        save_path = tempfile.mktemp() if save_path is None else save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        config = ExperimentConfig.default(
            f"error-heuristics",
            data_model=model_dict["glauber"],
            prior=prior_dict["erdosrenyi"],
            target=prior_dict["erdosrenyi"],
            metrics=["recon_error"],
            path=save_path,
            n_workers=n_workers,
            n_async_jobs=n_async_jobs,
            seed=seed,
        )

        config.metrics.recon_error.n_samples = n_samples_per_worker * ceil(
            n_workers / n_async_jobs
        )
        config.metrics.recon_error.reconstructor = [
            "bayesian",
            "peixoto",
            "correlation",
            "granger_causality",
            "transfer_entropy",
        ]
        config.metrics.recon_error.data_mcmc.n_sweeps = 10
        config.metrics.recon_error.graph_mcmc.n_sweeps = 10
        config.metrics.recon_error.measures = (
            "roc, posterior_similarity, accuracy, mean_error"
        )
        config.metrics.recon_error.reduction = "identity"

        config.resources.update(
            account=ACCOUNT,
            time=time,
            mem=f"{mem}G",
            cpus_per_task=config.n_workers,
            job_name=config.name,
            output=f"log/{config.name}.out",
        )
        config.lock()
        return config

    @staticmethod
    def all(time="24:00:00", n_workers=MAXNUMJOBS, mem=MAXMEMORY):
        return [
            ErrorHeuristicsScriptConfig.default(
                save_path=os.path.join(PATH_TO_DATA, "error-heuristics"),
                time=time,
                n_workers=n_workers,
                mem=mem,
            )
        ]
