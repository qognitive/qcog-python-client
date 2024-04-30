from __future__ import annotations

from enum import Enum
from typing import Type, TypedDict, TypeAlias, Any, Protocol

import pandas as pd


class Model(Enum):
    pauli = "pauli"
    ensemble = "ensemble"
    NOT_SET = "NOT_SET"


class EMPTY_DICTIONARY(TypedDict):
    pass


class Dataset(TypedDict):
    format: str
    source:str
    data: str
    project_guid: str


class WeightParams(TypedDict):
    """
    Definition of weight parameters
    must match the schema Input
    from orchestration API
    """
    learning_rate: float
    iterations: int
    optimization_method: str
    step_size: float
    first_moment_decay: float
    second_moment_decay: float
    epsilon: float


class FisherParams(TypedDict):
    """
    Definition of fisher parameters
    must match the schema Input
    from orchestration API
    """
    learning_rate: float


class StateParams(TypedDict):
    """
    Definition of states parameters
    must match the schema Input
    from orchestration API
    """
    state_method: str
    iterations: int
    learning_rate_axes: float
    fisher_axes_kwargs: FisherParams
    fisher_state_kwargs: FisherParams


NotRequiredWeightParams: TypeAlias = WeightParams | EMPTY_DICTIONARY
NotRequiredStateParams: TypeAlias = StateParams | EMPTY_DICTIONARY


class TrainingParameters(TypedDict):
    batch_size: int,
    num_passes: int,
    weight_optimization: NotRequiredWeightParams,
    get_states_extra: NotRequiredStateParams,


class TrainProtocol(Protocol):
    def train(
        self,
        batch_size: int,
        num_passes: int,
        weight_optimization: NotRequiredWeightParams,
        get_states_extra: NotRequiredStateParams,
    ) -> Any:  # NOTE: we could make this a generic
        raise NotImplementedError("Train class must implement train")


class InferenceProtocol(Protocol):
    def inference(
        self,
        data: pd.DataFrame,
        operators_to_forecast: list[Operator]
    ) -> pd.DataFrame:
        raise NotImplementedError("Inference class must implement inference")


Operator: TypeAlias = str | int


class PauliSchema(Protocol, TrainProtocol, InferenceProtocol):
    """
    Definition of Pauli parameters
    must match the "schema" validation
    from orchestration API
    """
    def __init__(
        self,
        operators: list[Operator],
        qbits: int,
        pauli_weight: int,
        sigma_sq: dict[str, float],
        sigma_sq_optimization: dict[str, float],
        seed: int,
        target_operator: list[Operator],
    ):
        raise NotImplementedError("Pauli class must implement init")


class EnsembleSchema(Protocol, TrainProtocol, InferenceProtocol):
    """
    Definition of Ensemble parameters
    must match the "schema" validation
    from orchestration API
    """
    def __init__(
        self,
        operators: list[str],
        dim: int,
        num_axes: int,
        sigma_sq: dict[str, float],
        sigma_sq_optimization: dict[str, float],
        seed: int,
        target_operator: list
    ):
        raise NotImplementedError("Pauli class must implement init")

    def inference(
        self,
        data: pd.DataFrame,
        operators_to_forecast: list[Operator]
    ) -> pd.DataFrame:
        """
        We could create a InferenceProtocol class since both Ensemble and Pauli
        share the same interface, but quantum is different. individual
        implementations is more future proof
        """
        raise NotImplementedError("Pauli class must implement inference")


class ValueMixin:
    model: Model

    @property
    def value(self) -> str:
        return self.model.value


class PauliModel(PauliSchema, ValueMixin):
    class payload(TypedDict):
        operators: list[Operator]
        qbits: int
        pauli_weight: int
        sigma_sq: dict[str, float]
        sigma_sq_optimization: dict[str, float]
        seed: int
        target_operator: list[Operator]

    def __init__(
        self,
        operators: list[Operator],
        qbits: int,
        pauli_weight: int,
        sigma_sq: dict[str, float],
        sigma_sq_optimization: dict[str, float],
        seed: int,
        target_operator: list[Operator],
    ):
        self.model = Model.pauli
        self.params = payload(
            operators,
            qbits,
            pauli_weight,
            sigma_sq,
            sigma_sq_optimization,
            seed,
            target_operator,
        )


class EnsembleModel(EnsembleSchema, ValueMixin):
    class payload(TypedDict):
        operators: list[Operator]
        dim: int
        num_axes: int
        sigma_sq: dict[str, float]
        sigma_sq_optimization: dict[str, float]
        seed: int
        target_operator: list[Operator]

    def __init__(
        self,
        operators: list[Operator],
        dim: int,
        num_axes: int,
        sigma_sq: dict[str, float],
        sigma_sq_optimization: dict[str, float],
        seed: int,
        target_operator: list[Operator],
    ):
        self._model = Model.ensemble
        self.params = payload(
            operators,
            dim,
            num_axes,
            sigma_sq,
            sigma_sq_optimization,
            seed,
            target_operator,
        )


class ModelClient:

    @classmethod
    def from_model_guid(
        cls: Type[ModelClient],
        guid: str,
        client: QcogClient | None = None,
        with_data: bool = False
    ) -> ModelClient:
        """
        Create a ModelClient instance from a model guid

        Parameters:
        -----------
        guid: str
            the model guid string
        client: QcogClient | None:
            optional application client

        Returns:
        --------
        ModelClient instance from trained model
        """
        qcog_client: QcogClient = (
            client if client is not None else QcogClient()
        )
        return cls(qcog_client).preloaded_model(guid, with_data=with_data)

    def __init__(
        self,
        client: QcogClient,
        version: str = "0.0.43"
    ):
        """
        Create a ModelClient instance from scratch

        Parameters:
        -----------
        client: QcogClient | None:
            optional application client
        version: str
            qcog package version to use. Default value is the OLDEST
            supported version

        Returns:
        --------
        ModelClient empty instance
        """
        self.model: Model = Model("NOT_SET")
        self.params: Interface = EMPTY_MODEL_PARAMS()
        self.client: QcogClient = client
        self.version: str = version
        self.dataset: dict = {}
        self.training_parameters: dict = {}
        self.trained_model: dict = {}

    def _rebuild_model_params(self) -> ModelClient:
        """
        For a pretrained model, create valid class
        instance state from the training parameters
        payload.

        Returns:
        --------
        ModelClient itself
        """
        self._model(self.training_parameters["model"])
        copy_params: dict = self.training_parameters[
            "parameters"
        ]["model"].copy()
        model_params: Interface = VALID_MODEL_PARAMS[self.model](
            **copy_params
        )

        return self._params(model_params)

    def _model(self, model: str) -> ModelClient:
        """
        For a fresh "to train" model, create valid class
        instance state from the str model.

        Parameters:
        -----------
        model: str
            A Model Enum valid str

        Returns:
        --------
        ModelClient itself
        """
        self.model = Model(model)
        return self

    def _params(self, params: Interface) -> ModelClient:
        """
        For a fresh "to train" model, create valid class
        instance state from the class params.

        Parameters:
        -----------
        params: Interface
            A specific class parametrization matching
            one from the enum

        Returns:
        --------
        ModelClient itself
        """
        if not params:
            raise ValueError("Invalid empty model parameters")
        if not isinstance(params, VALID_MODEL_PARAMS[self.model]):
            raise ValueError()
#            raise ValueError(f"Invalid type of model parameters got {type(params)} expected {VALID_MODEL_PARAMS[self.model])}")  # noqa: 503
        self.params = params
        return self

    def model_params(
        self,
        model: str,
        params: Interface,
    ) -> ModelClient:
        """
        For a fresh "to train" model, create valid class
        instance state from the class params following
        the model str.

        Parameters:
        -----------
        model: str
            A Model Enum valid str
        params: Interface
            A specific class parametrization matching
            one from the enum

        Returns:
        --------
        ModelClient itself
        """
        return self._model(model)._params(params)

    def preloaded_model(
        self, guid: str,
        with_data: bool = False,
    ) -> ModelClient:
        """
        For a pretrained model, create valid class
        instance state from the trained_model
        payload. Since the trained model
        has all references guids, this
        method can initialize everything.

        Parameters:
        -----------
        guid: str
            model guid
        with_data: bool
            if False, doesn't retrieve the training dataset

        Returns:
        --------
        ModelClient itself
        """
        self.trained_model = self.client.get(f"model/{guid}")

        training_parameters_guid = self.trained_model[
            'training_parameters_guid'
        ]
        dataset_guid = self.trained_model["dataset_guid"]
        # project_guid = self.trained_model["project_guid"]  # TODO review this
        self.version = self.trained_model[
            "training_package_location"
        ].split("packages/")[-1].split("-")[1]

        self.preloaded_training_parameters(
            training_parameters_guid
        )._rebuild_model_params()
        if with_data:
            return self.preloaded_data(dataset_guid)
        return self
