"""Qcog Python Client package."""


from .qcog import AsyncQcogClient, QcogClient  # noqa: F401
from .schema import (  # noqa: F401
    Model,  # noqa: F401
    ModelEnsembleParameters,
    ModelGeneralParameters,
    ModelPauliParameters,
)

EnsembleModel = Model.ensemble  # noqa: F401
PauliModel = Model.pauli  # noqa: F401
GeneralModel = Model.general  # noqa: F401
EnsembleSchema = ModelEnsembleParameters
PauliSchema = ModelPauliParameters
