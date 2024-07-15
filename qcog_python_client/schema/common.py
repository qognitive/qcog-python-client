"""Common types and classes used in the client."""

from __future__ import annotations

import enum
from typing import Any, Protocol, TypeAlias, TypedDict

import pandas as pd

from .generated_schema.models import (
    AdamOptimizationParameters,
    AnalyticOptimizationParameters,
    EIGHStateParameters,
    EIGSStateParameters,
    GradOptimizationParameters,
    GradStateParameters,
    LOBPCGFastStateParameters,
    Model,
    NPEIGHStateParameters,
    PowerIterStateParameters,
)

Operator: TypeAlias = str | int


class EmptyDictionary(TypedDict):
    """For TypedDict that are optional allow empty dict rather than use None."""

    pass


# TODO: define the state method in the backend so that the code generation
# can be done properly in the same way we already defined the optimization
# For now this will work as a interface in order to avoid breaking changes
# once the schema is defined.
class StateMethod(str, enum.Enum):
    """Enum definition for the state methods."""

    LOBPCG_FAST = "LOBPCG_FAST"
    POWER_ITER = "POWER_ITER"
    EIGS = "EIGS"
    EIGH = "EIGH"
    NP_EIGH = "NP_EIGH"
    LOBPCB = "LOBPCB"
    GRAD = "GRAD"


StateMethodModel: TypeAlias = StateMethod

WeightParams: TypeAlias = (
    AnalyticOptimizationParameters
    | AdamOptimizationParameters
    | GradOptimizationParameters
)

StateParams: TypeAlias = (
    LOBPCGFastStateParameters
    | PowerIterStateParameters
    | EIGHStateParameters
    | EIGSStateParameters
    | NPEIGHStateParameters
    | GradStateParameters
)


class InferenceParameters(TypedDict):
    """Inference Parameters."""

    operators_to_forecast: list[str] | None
    state_parameters: StateParams


NotRequiredWeightParams = WeightParams | EmptyDictionary
NotRequiredStateParams = StateParams | EmptyDictionary


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


Matrix: TypeAlias = list[list[int | float | Any]]
