"""Parameters for client requests.

Parameters are defined as Pydantic models, providing type hints and runtime
validation.
"""

from pydantic import BaseModel, Field
from typing import TypeAlias
import enum


############################################
# Optimization Parameters
############################################


class OptimizationMethod(str, enum.Enum):
    GRAD = "GRAD"
    ADAM = "ADAM"
    ANALYTIC = "ANALYTIC"


class GradOptimizationParameters(BaseModel):
    """Parameters for gradient descent optimization.
    ----------

    iterations: **int**
        This is how many gradient descent steps will be made before considering
        the state to have converged. Having more iterations and a lower
        learning rate corresponds to a better path through the energy
        landscape. So if you were to take 10 steps at `1e-3` learning rate that
        is more accurate, as we recompute our gradient 10 times, than a single
        step of `1e-2` learning rate. The recommended range is 3-10.

    learning_rate: **float**
        The learning rate for the gradient descent algorithm.
        This is fixed and does not decay during optimization.
        Recommended values are around `1e-3`.
    """

    optimization_method: OptimizationMethod = OptimizationMethod.GRAD
    iterations: int
    learning_rate: float


class AdamOptimizationParameters(BaseModel):
    """Parameters for Adam optimization.
    ----------
    iterations: **int**
        This is how many gradient descent steps will be made before considering
        the state to have converged. Having more iterations and a lower
        learning rate corresponds to a better path through the energy
        landscape. So if you were to take 10 steps at `1e-3` learning rate that
        is more accurate, as we recompute our gradient 10 times, than a single
        step of `1e-2` learning rate. The recommended range is 3-10.

    step_size: **float**, **default=1e-3**
        The learning rate for the ADAM algorithm. This is the starting point
        which it will decay from. fixed and does not decay during optimization.
        Recommended values are around `1e-3`.

    epsilon: **float**, **default=1e-8**
        A small parameter that is used for numerical stability in the ADAM
        algorithm. Recommended values are around `1e-8`.

    first_moment_decay: **float**, **default=0.9**
        The first moment decay, eseentially scaling a term that is linear in
        the gradient. Recommended values are around `0.9`.


    second_moment_decay: **float**, **default=0.999**
        The second moment decay, essentially scaling a term that is
        quadratic in the gradient. Recommended values are around 0.999.
    """

    optimization_method: OptimizationMethod = OptimizationMethod.ADAM
    iterations: int
    step_size: float = Field(default=1e-3)
    epsilon: float = Field(default=1e-8)
    first_moment_decay: float = Field(default=0.9)
    second_moment_decay: float = Field(default=0.999)


class AnalyticOptimizationParameters(BaseModel):
    """The analytic optimizer is a closed form solution to the internal
    weight matrix in our model. The analytic optimizer takes no parameters.
    ----------
    """
    optimization_method: OptimizationMethod = OptimizationMethod.ANALYTIC


############################################
# State Parameters
############################################


class StateMethod(str, enum.Enum):
    LOBPCG_FAST = "LOBPCG_FAST"
    POWER_ITER = "POWER_ITER"
    EIGS = "EIGS"
    EIGH = "EIGH"
    NP_EIGH = "NP_EIGH"
    LOBPCB = "LOBPCB"
    GRAD = "GRAD"


class LOBPCGFastStateParameters(BaseModel):
    """Parameters for the LOBPCG_FAST state method.
    ----------

    iterations: **int**
        The maximum number of iterations to run the LOBPCG algorithm for.
        You should think of this as a
        maximum number of internal iterations that are made to attempt the
        states to converge to within the tolerance. Both parameters work
        together, so if your tolerance is very large then you will need
        few iterations, so even if you set iteration count to 100 if it
        only takes 5 to converge to your tolerance only 5 iterations will
        be done. If your tolerance is very low but the iteration count is
        low then the algorithm will stop after the iteration count is
        reached, regardless of the tolerance. A good recommended range is
        5-20, noting that the more iterations you do the more accurate the
        state will be, but the more computationally expensive it will be.

    tol: **float**, **default=0.2**
        The tolerance for the LOBPCG algorithm. This is a relative tolerance
        and not an absolute one, so the units are arbitrary. Generally 0.2
        which is the default is a very loose tolerance. If you want to have
        the output from this method close to one of the more exact solvers
        then try `1e-4 -> 1e-8` for the tolerance. As per the above
        discussion you should also increase your iterations if you are
        decreasing your tolerance. The tolerance and iterations are more
        important for inference as the model will only pass over that data
        once.
    """

    iterations: int
    tol: float = Field(default=0.2)
    state_method: StateMethod = StateMethod.LOBPCG_FAST


class PowerIterStateParameters(BaseModel):
    """Parameters for the POWER_ITER state method.
    ----------

    iterations: **int**
        The maximum number of iterations to run the LOBPCG algorithm for.
        You should think of this as a
        maximum number of internal iterations that are made to attempt the
        states to converge to within the tolerance. Both parameters work
        together, so if your tolerance is very large then you will need
        few iterations, so even if you set iteration count to 100 if it
        only takes 5 to converge to your tolerance only 5 iterations will
        be done. If your tolerance is very low but the iteration count is
        low then the algorithm will stop after the iteration count is
        reached, regardless of the tolerance. A good recommended range is
        5-20, noting that the more iterations you do the more accurate the
        state will be, but the more computationally expensive it will be.

    tol: **float**, **default=0.2**
        The tolerance for the LOBPCG algorithm. This is a relative tolerance
        and not an absolute one, so the units are arbitrary. Generally 0.2
        which is the default is a very loose tolerance. If you want to have
        the output from this method close to one of the more exact solvers
        then try `1e-4 -> 1e-8` for the tolerance. As per the above
        discussion you should also increase your iterations if you are
        decreasing your tolerance. The tolerance and iterations are more
        important for inference as the model will only pass over that data
        once.

    max_eig_iter: **int**, **default=5**
        The number of iterations to execute to find the largest eigenvalue,
        which is used as a parameter in a spectral shift to find the
        smallest eigenvalue. 5 is generally a good default.
    """

    iterations: int
    tol: float = Field(default=0.2)
    max_eig_iter: int = Field(default=5)
    state_method: StateMethod = StateMethod.POWER_ITER


class EIGHStateParameters(BaseModel):
    """EIGH state method takes no parameters.
    ----------
    """

    state_method: StateMethod = StateMethod.EIGH


class EIGSStateParameters(BaseModel):
    """EIGS state method takes no parameters.
    ----------
    """

    state_method: StateMethod = StateMethod.EIGS


class NPEIGHStateParameters(BaseModel):
    """NP_EIGH state method takes no parameters.
    ----------
    """

    state_method: StateMethod = StateMethod.NP_EIGH


class LOBPCGStateParameters(BaseModel):
    """Parameters for the LOBPCG state method.
    ----------

    iterations: **int**
        You should think of this as a maximum number of internal iterations
        that are made to attempt the states to converge to within the
        tolerance. Both parameters work together, so if your tolerance
        is very large then you will need few iterations, so even if you
        set iteration count to 100 if it only takes 5 to converge to your
        tolerance only 5 iterations will be done. If your tolerance is very
        low but the iteration count is low then the algorithm will stop
        after the iteration count is reached, regardless of the tolerance.
        A good recommended range is 5-20, noting that the more iterations
        you do the more accurate the state will be, but the more
        computationally expensive it will be.

    tol: **float**, **default=0.2**
        This is a relative tolerance and not an absolute one, so the units are
        arbitrary. Generally 0.2 which is the default is a very loose
        tolerance. If you want to have the output from this method close to
        one of the more exact solvers then try `1e-4 -> 1e-8` for the
        tolerance. As per the above discussion you should also increase your
        iterations if you are decreasing your tolerance. The tolerance and
        iterations are more important for inference as the model will only
        pass over that data once.
    """

    iterations: int
    tol: float = Field(default=0.2)
    state_method: StateMethod = StateMethod.LOBPCB


class GradStateParameters(BaseModel):
    """Parameters for gradient descent optimization.
    ----------

    iterations: **int**
        This is how many gradient descent steps will be made before considering
        the state to have converged. Having more iterations and a lower
        learning rate corresponds to a better path through the energy
        landscape. So if you were to take 10 steps at `1e-3` learning rate
        that is more accurate, as we recompute our gradient 10 times, than a
        single step of `1e-2` learning rate. The recommended range is 3-10.

    learning_rate: **float**
        The learning rate for the gradient descent algorithm.
        This is fixed and does not decay during optimization. Recommended
        values are around `1e-3`.
    """

    iterations: int
    learning_rate: float
    state_method: StateMethod = StateMethod.GRAD


WeightParams: TypeAlias = (
    GradOptimizationParameters
    | AdamOptimizationParameters
    | AnalyticOptimizationParameters
)

StateParams: TypeAlias = (
    LOBPCGFastStateParameters
    | PowerIterStateParameters
    | EIGHStateParameters
    | EIGSStateParameters
    | NPEIGHStateParameters
    | GradStateParameters
)
