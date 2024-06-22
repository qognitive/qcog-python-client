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
    """
    Pauli model class "prototype"
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


class PauliSchema(PauliProtocol, TrainProtocol, InferenceProtocol):
    """Pauli-model specific parameters.

    Parameters
    ----------
    operators : list[Operator]
        List of operators to be used in the model. These should be the names
        of the columns in the dataframe for the dataset being trained.
    qbits : int
        Number of qubits in the model, corresponding to the dimensionality of
        our internal state.
    pauli_weight : int
        This corresponds to the sparsity of our representation in our internal
        state. 1 is the most sparse and the maximum here is equal to the qbits
        specified, which would be a full dense representation.
    sigma_sq : dict[str, float]
        Dictionary of scaling factors where the keys are the operators and the
        values are the scaling factors. These are in the form of 1/sigma so a
        small sigma will increase the weight of that operator in the model.
    seed : int
        A random seed which is used to initialize the model, you can set this
        in order to increase reproducibility.
    """
    pass


class PauliModel(PauliProtocol, ValueMixin):
    """
    client side schema implementation
    """

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
