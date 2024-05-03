from __future__ import annotations

from typing import TypedDict, Protocol

from .common import (
    Model,
    Operator,
    TrainProtocol,
    InferenceProtocol,
    ValueMixin,
)


class PauliProtocol(Protocol):
    def __init__(
        self,
        operators: list[Operator],
        qbits: int,
        pauli_weight: int,
        sigma_sq: dict[str, float],
        sigma_sq_optimization: dict[str, float],
        seed: int,
        target_operators: list[Operator],
    ):
        raise NotImplementedError("Pauli class must implement init")


class PauliSchema(PauliProtocol, TrainProtocol, InferenceProtocol):
    pass


class PauliModel(PauliProtocol, ValueMixin):

    model = Model.pauli

    class payload(TypedDict):
        operators: list[Operator]
        qbits: int
        pauli_weight: int
        sigma_sq: dict[str, float]
        sigma_sq_optimization_kwargs: dict[str, float]
        seed: int
        target_operators: list[Operator]

    def __init__(
        self,
        operators: list[Operator],
        qbits: int,
        pauli_weight: int,
        sigma_sq: dict[str, float],
        sigma_sq_optimization: dict[str, float],
        seed: int,
        target_operators: list[Operator],
    ):
        self.params = self.payload(
            operators=operators,
            qbits=qbits,
            pauli_weight=pauli_weight,
            sigma_sq=sigma_sq,
            sigma_sq_optimization_kwargs=sigma_sq_optimization,
            seed=seed,
            target_operators=target_operators,
        )
