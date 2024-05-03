from __future__ import annotations

from enum import Enum
from typing import TypedDict, TypeAlias, Any, Protocol

import numpy as np
import pandas as pd


class Model(Enum):
    """
    List of available models
    """
    pauli = "pauli"
    ensemble = "ensemble"


class EMPTY_DICTIONARY(TypedDict):
    """
    For TypedDict that are optional
    allow empty dict rather than use None
    """
    pass


class Dataset(TypedDict):
    format: str
    source: str
    data: str
    project_guid: str


class WeightParams(TypedDict):
    learning_rate: float
    iterations: int
    optimization_method: str
    step_size: float
    first_moment_decay: float
    second_moment_decay: float
    epsilon: float


class FisherParams(TypedDict):
    learning_rate: float


class StateParams(TypedDict):
    state_method: str
    iterations: int
    learning_rate_axes: float
    # TODO we should remove the "kwargs"
    # from the names and favor explicit
    # definittions
    fisher_axes_kwargs: FisherParams
    fisher_state_kwargs: FisherParams


class InferenceParameters(TypedDict):
    operators_to_forecast: list[str] | None
    states: np.ndarray | None
    return_states: bool
    # TODO we should remove the "kwargs"
    # from the names and favor explicit
    # definittions
    kwargs: dict


NotRequiredWeightParams: TypeAlias = WeightParams | EMPTY_DICTIONARY
NotRequiredStateParams: TypeAlias = StateParams | EMPTY_DICTIONARY


class TrainingParameters(TypedDict):
    batch_size: int
    num_passes: int
    # TODO we should remove the "kwargs"
    # from the names and favor explicit
    # definittions
    weight_optimization_kwargs: NotRequiredWeightParams
    state_kwargs: NotRequiredStateParams


class TrainProtocol(Protocol):
    """
    Train method "prototype"
    """
    def train(
        self,
        batch_size: int,
        num_passes: int,
        weight_optimization: NotRequiredWeightParams,
        get_states_extra: NotRequiredStateParams,
    ) -> Any:  # NOTE: we could make this a generic
        raise NotImplementedError("Train class must implement train")


class InferenceProtocol(Protocol):
    """
    Inference method "prototype"
    """
    def inference(
        self,
        data: pd.DataFrame,
        parameters: InferenceParameters,
    ) -> pd.DataFrame:
        raise NotImplementedError("Inference class must implement inference")


Operator: TypeAlias = str | int


class ValueMixin:
    """
    Utility mixin for the client Models
    """
    model: Model

    @property
    def value(self) -> str:
        return self.model.value
