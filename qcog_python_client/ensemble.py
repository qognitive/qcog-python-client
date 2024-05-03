from __future__ import annotations

from typing import TypedDict, Protocol

from .common import (
    Model,
    Operator,
    TrainProtocol,
    InferenceProtocol,
    ValueMixin,
)


class EnsembleProtocol(Protocol):
    """
    Ensemble model class "prototype"
    """
    def __init__(
        self,
        operators: list[Operator],
        dim: int,
        num_axes: int,
        sigma_sq: dict[str, float],
        sigma_sq_optimization: dict[str, float],
        seed: int,
        target_operators: list[Operator],
    ):
        raise NotImplementedError("Ensemble class must implement init")


class EnsembleSchema(EnsembleProtocol, TrainProtocol, InferenceProtocol):
    """
    Schema definition meant to be used externally
    """
    pass


class EnsembleModel(EnsembleProtocol, ValueMixin):
    """
    client side schema implementation
    """

    model = Model.pauli

    class payload(TypedDict):
        operators: list[Operator]
        dim: int
        num_axes: int
        sigma_sq: dict[str, float]
        sigma_sq_optimization_kwargs: dict[str, float]
        seed: int
        target_operators: list[Operator]

    def __init__(
        self,
        operators: list[Operator],
        dim: int,
        num_axes: int,
        sigma_sq: dict[str, float],
        sigma_sq_optimization: dict[str, float],
        seed: int,
        target_operators: list[Operator],
    ):
        self.params = self.payload(
            operators=operators,
            dim=dim,
            num_axes=num_axes,
            sigma_sq=sigma_sq,
            sigma_sq_optimization_kwargs=sigma_sq_optimization,
            seed=seed,
            target_operators=target_operators,
        )
