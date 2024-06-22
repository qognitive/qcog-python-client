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
    """Ensemble-model specific parameters.

    Parameters
    ----------
    operators : list[Operator]
        List of operators to be used in the model. These should be the names
        of the columns in the dataframe for the dataset being trained.
    dim : int
        The dimensional size of our internal state.
    num_axes : int
        This corresponds to the sparsity of our representation in our internal
        state, and it can be thought of as how many basis vectors we are decomposing
        our internal state into. 1 would make our internal state of low rank and
        the maximum here is equal to dim^2, which would be a full dense representation.
    sigma_sq : dict[str, float]
        Dictionary of scaling factors where the keys are the operators and the
        values are the scaling factors. These are in the form of 1/sigma so a
        small sigma will increase the weight of that operator in the model.
    seed : int
        A random seed which is used to initialize the model, you can set this
        in order to increase reproducibility.
    """
    pass


class EnsembleModel(EnsembleProtocol, ValueMixin):
    """
    client side schema implementation
    """

    model = Model.ensemble

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
