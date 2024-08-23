"""Wandb Monitor implementation."""

import os

import wandb

from .interface import Monitor

WANDB_DEFAULT_PROJECT = "qcognitive-dev"


class WandbMonitor(Monitor):
    """Wandb Monitor implementation."""

    def init(
        self,
        api_key: str | None = None,
        project: str = WANDB_DEFAULT_PROJECT,
        parameters: dict | None = None,
    ) -> None:
        """Initialize the Wandb Monitor."""
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
        )

    def log(self, data: dict) -> None:
        """Log data to Wandb."""
        wandb.log(data)

    def close(self) -> None:
        """Close the Wandb Monitor."""
        wandb.finish()
