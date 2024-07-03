"""Common types and classes used in the client."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, TypeAlias, TypedDict

import pandas as pd
from typing_extensions import NotRequired

from .parameters import StateParams, WeightParams

Operator: TypeAlias = str | int


class Model(Enum):
    """List of available models."""

    pauli = "pauli"
    ensemble = "ensemble"
    general = "general"


class EmptyDictionary(TypedDict):
    """For TypedDict that are optional allow empty dict rather than use None."""

    pass


class Dataset(TypedDict):
    """Dataset Parameters."""

    format: str
    source: str
    data: str


class FisherParams(TypedDict):
    """Fisher Parameters."""

    learning_rate: NotRequired[float]
    init_update_n: int
    update_frequency: int
    adjust_lr: bool
    average_over_axis: NotRequired[int]
    use_hessian: bool


class InferenceParameters(TypedDict):
    """Inference Parameters."""

    operators_to_forecast: list[str] | None
    state_parameters: StateParams


NotRequiredWeightParams: TypeAlias = WeightParams | EmptyDictionary
NotRequiredStateParams: TypeAlias = StateParams | EmptyDictionary


class TrainingParameters(TypedDict):
    """Training Parameters.

    Dictionary of training parameters.

    ----------

    batch_size : int
        Number of samples to use in each training batch.

    num_passes : int
        Number of passes through the dataset.

    weight_optimization_kwargs : NotRequiredWeightParams
        Weight optimization parameters.

    state_kwargs : NotRequiredStateParams
        State optimization parameters.

    """

    batch_size: int
    num_passes: int
    weight_optimization_kwargs: NotRequiredWeightParams
    state_kwargs: NotRequiredStateParams


class AsyncTrainProtocol(Protocol):
    """Train method "prototype"."""

    async def train(  # noqa: D102
        self,
        batch_size: int,
        num_passes: int,
        weight_optimization: NotRequiredWeightParams,
        get_states_extra: NotRequiredStateParams,
    ) -> Any:  # NOTE: we could make this a generic
        raise NotImplementedError("Train class must implement train")


class AsyncInferenceProtocol(Protocol):
    """Inference method "prototype"."""

    async def inference(  # noqa: D102
        self,
        data: pd.DataFrame,
        parameters: InferenceParameters,
    ) -> pd.DataFrame:
        raise NotImplementedError("Inference class must implement inference")


class TrainProtocol(Protocol):
    """Train method "prototype"."""

    def train(  # noqa: D102
        self,
        batch_size: int,
        num_passes: int,
        weight_optimization: NotRequiredWeightParams,
        get_states_extra: NotRequiredStateParams,
    ) -> Any:  # NOTE: we could make this a generic
        raise NotImplementedError("Train class must implement train")


class InferenceProtocol(Protocol):
    """Inference method "prototype"."""

    def inference(  # noqa: D102
        self,
        data: pd.DataFrame,
        parameters: InferenceParameters,
    ) -> pd.DataFrame:
        raise NotImplementedError("Inference class must implement inference")


class ValueMixin:
    """Utility mixin for the client Models."""

    model: Model

    @property
    def value(self) -> str:
        """Return the model value."""
        return self.model.value
