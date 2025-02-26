"""Beaker entrypoints for 20250122_worldcover_sampling."""

from beaker import (
    Beaker,
    Constraints,
    ExperimentSpec,
    Priority,
    TaskResources,
    TaskSpec,
)
from beaker.services.experiment import ExperimentClient


def launch_worldcover_job(
    beaker_image: str,
    workspace: str = "ai2/gabi-workspace",
    budget: str = "ai2/d5",
    clusters: list[str] = [
        "ai2/jupiter-cirrascale-2",
        "ai2/saturn-cirrascale",
        "ai2/ceres-cirrascale",
    ],
) -> None:
    """Launch worldcover job."""
    description = "worldcover_1km_sampling"
    beaker = Beaker.from_env(default_workspace=workspace)

    result_path = "/outputs"

    spec = ExperimentSpec(
        budget=budget,
        description=description,
        tasks=[
            TaskSpec.new(
                name=description,
                beaker_image=beaker_image,
                command=["python", "-u", "20250122_worldcover_sampling.py"],
                constraints=Constraints(cluster=clusters),
                resources=TaskResources(gpu_count=0),
                priority=Priority.normal,
                preemptible=True,
                result_path=result_path,
            )
        ],
    )
    experiment = beaker.experiment.create(description, spec)
    experiment_client = ExperimentClient(beaker)
    print(f"Experiment created: {experiment.id}: {experiment_client.url(experiment)}")
