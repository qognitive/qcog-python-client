from __future__ import annotations

from enum import Enum
from typing import TypedDict, TypeAlias, Any, Protocol

import pandas as pd


class Model(Enum):
    pauli = "pauli"
    ensemble = "ensemble"
    NOT_SET = "NOT_SET"


class EMPTY_DICTIONARY(TypedDict):
    pass


class Dataset(TypedDict):
    format: str
    source: str
    data: str
    project_guid: str


class WeightParams(TypedDict):
    """
    Definition of weight parameters
    must match the schema Input
    from orchestration API
    """
    learning_rate: float
    iterations: int
    optimization_method: str
    step_size: float
    first_moment_decay: float
    second_moment_decay: float
    epsilon: float


class FisherParams(TypedDict):
    """
    Definition of fisher parameters
    must match the schema Input
    from orchestration API
    """
    learning_rate: float


class StateParams(TypedDict):
    """
    Definition of states parameters
    must match the schema Input
    from orchestration API
    """
    state_method: str
    iterations: int
    learning_rate_axes: float
    fisher_axes_kwargs: FisherParams
    fisher_state_kwargs: FisherParams


NotRequiredWeightParams: TypeAlias = WeightParams | EMPTY_DICTIONARY
NotRequiredStateParams: TypeAlias = StateParams | EMPTY_DICTIONARY


class TrainingParameters(TypedDict):
    batch_size: int
    num_passes: int
    # weight_optimization: NotRequiredWeightParams
    # get_states_extra: NotRequiredStateParams
    weight_optimization_kwargs: NotRequiredWeightParams
    state_kwargs: NotRequiredStateParams


class TrainProtocol(Protocol):
    def train(
        self,
        batch_size: int,
        num_passes: int,
        weight_optimization: NotRequiredWeightParams,
        get_states_extra: NotRequiredStateParams,
    ) -> Any:  # NOTE: we could make this a generic
        raise NotImplementedError("Train class must implement train")


class InferenceProtocol(Protocol):
    def inference(
        self,
        data: pd.DataFrame,
        operators_to_forecast: list[Operator]
    ) -> pd.DataFrame:
        raise NotImplementedError("Inference class must implement inference")


Operator: TypeAlias = str | int


class PauliSchema(TrainProtocol, InferenceProtocol):
    """
    Definition of Pauli parameters
    must match the "schema" validation
    from orchestration API
    """
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


class EnsembleSchema(TrainProtocol, InferenceProtocol):
    """
    Definition of Ensemble parameters
    must match the "schema" validation
    from orchestration API
    """
    def __init__(
        self,
        operators: list[str],
        dim: int,
        num_axes: int,
        sigma_sq: dict[str, float],
        sigma_sq_optimization: dict[str, float],
        seed: int,
        target_operators: list
    ):
        raise NotImplementedError("Pauli class must implement init")

    def inference(
        self,
        data: pd.DataFrame,
        operators_to_forecast: list[Operator]
    ) -> pd.DataFrame:
        """
        We could create a InferenceProtocol class since both Ensemble and Pauli
        share the same interface, but quantum is different. individual
        implementations is more future proof
        """
        raise NotImplementedError("Pauli class must implement inference")


class ValueMixin:
    model: Model

    @property
    def value(self) -> str:
        return self.model.value


class PauliModel(PauliSchema, ValueMixin):
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
        self.model = Model.pauli
        self.params = self.payload(
            operators=operators,
            qbits=qbits,
            pauli_weight=pauli_weight,
            sigma_sq=sigma_sq,
            sigma_sq_optimization_kwargs=sigma_sq_optimization,
            seed=seed,
            target_operators=target_operators,
        )


class EnsembleModel(EnsembleSchema, ValueMixin):
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
        self.model = Model.ensemble
        self.params = self.payload(
            operators=operators,
            dim=dim,
            num_axes=num_axes,
            sigma_sq=sigma_sq,
            sigma_sq_optimization_kwargs=sigma_sq_optimization,
            seed=seed,
            target_operators=target_operators,
        )
