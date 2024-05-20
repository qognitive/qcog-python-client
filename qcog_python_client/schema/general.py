from __future__ import annotations

from typing import TypedDict, Protocol

from .common import (
    Model,
    Operator,
    TrainProtocol,
    InferenceProtocol,
    ValueMixin,
)


class GeneralProtocol(Protocol):
    """
    General model class "prototype"
    """
    def __init__(
        self,
        operators: list[Operator],
        dims: int,
        sigma_sq: dict[str, float],
        seed: int,
        target_operators: list[Operator],
    ):
        raise NotImplementedError("General class must implement init")


class GeneralSchema(GeneralProtocol, TrainProtocol, InferenceProtocol):
    """
    Schema definition meant to be used externally
    """
    pass


class GeneralModel(GeneralProtocol, ValueMixin):
    """
    client side schema implementation
    """

    model = Model.general

    class payload(TypedDict):
        operators: list[Operator]
        dims: int
        sigma_sq: dict[str, float]
        seed: int
        target_operators: list[Operator]

    def __init__(
        self,
        operators: list[Operator],
        dims: int,
        sigma_sq: dict[str, float],
        sigma_sq_optimization: dict[str, float],
        seed: int,
        target_operators: list[Operator],
    ):
        self.params = self.payload(
            operators=operators,
            dims=dims,
            sigma_sq=sigma_sq,
            seed=seed,
            target_operators=target_operators,
        )
