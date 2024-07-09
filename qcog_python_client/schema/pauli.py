"""Pauli model schema definition."""

# from __future__ import annotations

# from typing import Protocol, TypedDict

# from .common import (
#     Operator,
#     ValueMixin,
# )
# from .generated_schema.models import Model


# class PauliProtocol(Protocol):
#     """Pauli model class "prototype"."""

#     def __init__(  # noqa: D107
#         self,
#         operators: list[Operator],
#         qbits: int,
#         pauli_weight: int,
#         sigma_sq: dict[str, float],
#         sigma_sq_optimization: dict[str, float],
#         seed: int,
#         target_operators: list[Operator],
#     ):
#         raise NotImplementedError("Pauli class must implement init")


# class PauliModel(PauliProtocol, ValueMixin):
#     """client side schema implementation."""

#     model = Model.pauli

#     class payload(TypedDict):  # noqa: N801, D106
#         operators: list[Operator]
#         qbits: int
#         pauli_weight: int
#         sigma_sq: dict[str, float]
#         sigma_sq_optimization_kwargs: dict[str, float]
#         seed: int
#         target_operators: list[Operator]

#     def __init__(  # noqa: D107
#         self,
#         operators: list[Operator],
#         qbits: int,
#         pauli_weight: int,
#         sigma_sq: dict[str, float],
#         sigma_sq_optimization: dict[str, float],
#         seed: int,
#         target_operators: list[Operator],
#     ):
#         self.params = self.payload(
#             operators=operators,
#             qbits=qbits,
#             pauli_weight=pauli_weight,
#             sigma_sq=sigma_sq,
#             sigma_sq_optimization_kwargs=sigma_sq_optimization,
#             seed=seed,
#             target_operators=target_operators,
#         )
