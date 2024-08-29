"""Wandb Monitor implementation."""

import os

import wandb

from .interface import Monitor

WANDB_DEFAULT_PROJECT = "qognitive-dev"


class WandbMonitor(Monitor):
    """Wandb Monitor implementation."""

    def init(  # noqa: D417   # Complains about parameters description not being present
        self,
        api_key: str | None = None,
        project: str = WANDB_DEFAULT_PROJECT,
        parameters: dict | None = None,
        labels: list[str] | None = None,
        trace_name: str | None = None,
    ) -> None:
        """Initialize the Wandb Monitor.

        Parameters
        ----------
            api_key : str | None
                Wandb API key.
            project : str | None
                Name of the project.
            parameters : dict | None
                Hyperparameters to be logged.
            labels : list | None
                Tags to be associated with the project.

        Raises
        ------
            ValueError: If the Wandb API key is not provided.

        """
        key = api_key or os.getenv("WANDB_API_KEY")

        if not key:
            raise ValueError(
                "Wandb API key is required. Please provide a key, ether as an argument or as an environment variable -> WANDB_API_KEY"  # noqa
            )
        wandb.login(
            anonymous="never",
            key=key,
        )

        wandb.init(
            project=project,
            config=parameters,
            tags=labels,
            name=trace_name,
        )

    def log(self, data: dict) -> None:  # noqa: D417   # Complains about parameters description not being present
        """Log data to Wandb.

        Parameters
        ----------
            data : dict Data to be logged.

        """
        wandb.log(data)

    def close(self) -> None:
        """Close the Wandb monitor."""
        wandb.finish()
