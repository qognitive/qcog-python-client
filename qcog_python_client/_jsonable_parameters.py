"""jsonable_parameters.

Private module, provides functionality to convert all the parameters
to a python dictionary while maintaining compatibility with the
current API schema.
"""

# Shamefull function to create a jsonable dict and keep compatibility
# with the current API schema. This will be removed once the schema
# is defined and updated.
from typing import Any

from pydantic import BaseModel

from qcog_python_client.schema.common import TrainingParameters
from qcog_python_client.schema.parameters import OptimizationMethod, StateMethod


def jsonable_parameters(params: TrainingParameters) -> dict:
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
        optimization_method: OptimizationMethod

        model_config = {"extra": "ignore"}

    ExpectedWeightParams.model_rebuild()

    class ExpectedStateParams(BaseModel):
        state_method: StateMethod
        iterations: int = 0
        learning_rate_axes: float = 0.0

        model_config = {"extra": "ignore"}

    ExpectedStateParams.model_rebuild()

    state_kwargs = params["state_kwargs"]
    weight_kwargs = params["weight_optimization_kwargs"]

    # Define empty dicts for the parameters
    weight_params = {}
    state_params = {}

    # If an object is actually passed
    if state_kwargs:
        # Dump the actual schema (based on the documentation)
        state_dict = state_kwargs.model_dump()
        # Parse it in order to only keep the actual expected parameters.
        # This step is necessary to have compatibility with the current
        # API schema.
        state_params = ExpectedStateParams.model_validate(state_dict).model_dump()

    # Repeate same process for the weight optimization parameters
    if weight_kwargs:
        weight_dict = weight_kwargs.model_dump()
        weight_params = ExpectedWeightParams.model_validate(weight_dict).model_dump()

    retval: dict[str, Any] = {
        "batch_size": params["batch_size"],
        "num_passes": params["num_passes"],
    }

    if weight_params:
        retval["weight_optimization_kwargs"] = weight_params

    if state_params:
        retval["state_kwargs"] = state_params

    return retval
