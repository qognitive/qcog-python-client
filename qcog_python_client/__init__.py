"""Qcog Python Client package."""

from .qcog import AsyncQcogClient, QcogClient
from .schema import (
    Model,
    ModelEnsembleParameters,
    ModelGeneralParameters,
    ModelPauliParameters,
)

EnsembleModel = Model.ensemble
PauliModel = Model.pauli
GeneralModel = Model.general
EnsembleSchema = ModelEnsembleParameters
PauliSchema = ModelPauliParameters
