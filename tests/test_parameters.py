"""Test validity of state and weight parameters."""

from pydantic import BaseModel

from qcog_python_client.qcog._jsonable_parameters import jsonable_train_parameters
from qcog_python_client.schema import (
    AdamOptimizationParameters,
    AnalyticOptimizationParameters,
    EIGHStateParameters,
    EIGSStateParameters,
    GradOptimizationParameters,
    LOBPCGFastStateParameters,
    LOBPCGStateParameters,
    NPEIGHStateParameters,
    PowerIterStateParameters,
)
from qcog_python_client.schema.common import TrainingParameters

# This test is meant to ensure that all the pydantic models,
# once parsed in the jsonable_parameters function, dont't produce
# any validation errors with the ExpectedWeightParams and ExpectedStateParams

weight_parameters: tuple = (
    GradOptimizationParameters(iterations=10, learning_rate=1e-3),
    # Adam override all defaults
    AdamOptimizationParameters(
        iterations=10,
        step_size=1e03,
        epsilon=1e-3,
        first_moment_decay=-0.3,
        second_moment_decay=0.8888,
    ),
    # Adam keep defaults
    AdamOptimizationParameters(iterations=10),
    AnalyticOptimizationParameters(),
)

state_parameters: tuple = (
    # LOBPCG override all defaults
    LOBPCGFastStateParameters(iterations=10, tol=0.1),
    # LOBPCG keep defaults
    LOBPCGFastStateParameters(iterations=10),
    # Power State override all defaults
    PowerIterStateParameters(iterations=10, tol=0.1, max_eig_iter=10),
    # Power State keep defaults
    PowerIterStateParameters(iterations=10),
    # LOBPCGS override all defaults
    LOBPCGStateParameters(iterations=10, tol=0.1),
    # LOBPCGS keep defaults
    LOBPCGStateParameters(iterations=10),
    # No parameters optimizations
    EIGHStateParameters(),
    EIGSStateParameters(),
    NPEIGHStateParameters(),
)


def test_parameters_api_adapting():
    """Test API parameters adapter.

    Currently some of the parameters are not taken in consideration by the
    API. This test is to ensure that the parameters are correctly adapted,
    the default values are correctly preserved and the overriden values are
    correctly set.
    """
    for weight_param in weight_parameters:
        for state_param in state_parameters:
            assert isinstance(weight_param, BaseModel)
            assert isinstance(state_param, BaseModel)

            params = jsonable_train_parameters(
                TrainingParameters(
                    batch_size=1000,
                    num_passes=10,
                    weight_optimization_kwargs=weight_param,
                    state_kwargs=state_param,
                )
            )

            # For each parameter, check if there is a corresponding paramter
            # in the jsonable_parameters dictionary. If there is, make sure
            # That the value is the same as the on in the passed parameter
            for k, v in weight_param.model_dump().items():
                param = params.get(k)

                if param:
                    assert params[param] == v

            # Do the same for the state parameters
            for k, v in state_param.model_dump().items():
                param = params.get(k)

                if param:
                    assert params[param] == v
