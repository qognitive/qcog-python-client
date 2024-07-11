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

from pydantic import BaseModel

from qcog_python_client.schema.common import (
    InferenceParameters,
    StateMethodModel,
    TrainingParameters,
)

from .schema import OptimizationMethodModel

# from qcog_python_client.schema.parameters import OptimizationMethod, StateMethod


def jsonable_train_parameters(params: TrainingParameters) -> dict:
    # Expected params for the API.
    # This will eventually change when the
    # schema is defined and updates
    class ExpectedWeightParams(BaseModel):
        learning_rate: float = 0.0
        iterations: int = 0
        step_size: float = 0.0
        first_moment_decay: float = 0.0
        second_moment_decay: float = 0.0
        epsilon: float = 0.0
        optimization_method: OptimizationMethodModel

        model_config = {"extra": "ignore"}

    ExpectedWeightParams.model_rebuild()

    class ExpectedStateParams(BaseModel):
        state_method: StateMethodModel
        iterations: int = 0
        learning_rate_axes: float = 0.0

        model_config = {"extra": "ignore"}

    ExpectedStateParams.model_rebuild()

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
        # Dump the actual schema (based on the documentation)
        state_dict = state_kwargs.model_dump()
        # Parse it in order to only keep the actual expected parameters.
        # This step is necessary to have compatibility with the current
        # API schema.
        state_params = ExpectedStateParams.model_validate(state_dict).model_dump()
        # Enums are not serializable, so we need to convert them to strings
        state_params["state_method"] = enum_serializable(state_params["state_method"])

    # Repeate same process for the weight optimization parameters
    if weight_kwargs:
        weight_dict = weight_kwargs.model_dump()
        weight_params = ExpectedWeightParams.model_validate(weight_dict).model_dump()
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
