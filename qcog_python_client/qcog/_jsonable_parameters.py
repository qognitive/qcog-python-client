"""jsonable_parameters.

Private module, provides functionality to convert all the parameters
to a python dictionary while maintaining compatibility with the
current API schema.
"""

# Shamefull function to create a jsonable dict and keep compatibility
# with the current API schema. This will be removed once the schema
# is defined and updated.
import enum
from typing import Any

from qcog_python_client.schema.common import (
    InferenceParameters,
    PytorchTrainingParameters,
    TrainingParameters,
)


def jsonable_train_parameters(
    params: TrainingParameters | PytorchTrainingParameters,
) -> dict:
    # PytorchTrainingParameters represent a dictionary with any key and value.
    if isinstance(params, PytorchTrainingParameters):
        return params.model_dump()

    state_kwargs = params["state_kwargs"]
    weight_kwargs = params["weight_optimization_kwargs"]

    # Define empty dicts for the parameters
    weight_params = {}
    state_params = {}

    def enum_serializable(e: enum.Enum | str) -> str:
        """Make an enum serializable by converting it to a string."""
        if not isinstance(e, (enum.Enum, str)):
            raise ValueError(f"Expected enum or string, got {type(e)}")

        if isinstance(e, enum.Enum):
            return str(e.value)
        return e

    # If an object is actually passed
    if state_kwargs:
        state_params = state_kwargs.model_dump()
        # Enums are not serializable, so we need to convert them to strings
        state_params["state_method"] = enum_serializable(state_params["state_method"])

    # Repeate same process for the weight optimization parameters
    if weight_kwargs:
        weight_params = weight_kwargs.model_dump()
        weight_params["optimization_method"] = enum_serializable(
            weight_params["optimization_method"]
        )

    retval: dict[str, Any] = {
        "batch_size": params["batch_size"],
        "num_passes": params["num_passes"],
    }

    if weight_params:
        retval["weight_optimization_kwargs"] = weight_params

    if state_params:
        retval["state_kwargs"] = state_params

    return retval


def jsonable_inference_parameters(params: InferenceParameters) -> dict:
    parameters = {}

    state_parameters = params.get("state_parameters")

    if state_parameters:
        parameters = params["state_parameters"].model_dump()

    return parameters
