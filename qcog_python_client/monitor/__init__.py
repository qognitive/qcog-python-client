from typing import Literal, TypeAlias

from ._wandb import WandbMonitor

Service: TypeAlias = Literal["wandb"]


def get_monitor(service: Service) -> WandbMonitor:
    """Return the monitoring service."""
    if service == "wandb":
        return WandbMonitor()

    raise ValueError(f"Unknown service: {service}")
