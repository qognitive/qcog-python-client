"""Qcog Python Client package."""

from pydantic import BaseModel

# Import before everything else in order to patch the BaseModel class
BaseModel.model_config = {"protected_namespaces": ()}

from .qcog import AsyncQcogClient, QcogClient  # noqa: E402
from .schema import (  # noqa: E402
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
