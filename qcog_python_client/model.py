from __future__ import annotations

import base64
from enum import Enum
from typing import Type, TypedDict, TypeAlias, Any
from typing_extensions import NotRequired
import pandas as pd
#  import numpy as np

from .client import QcogClient


class Model(Enum):
    pauli = "pauli"
    ensemble = "ensemble"
    NOT_SET = "NOT_SET"


class EMPTY_MODEL_PARAMS(TypedDict):
    """
    Typed empty dictionary as a default a value
    You can think or it as a Typed "None"
    """
    pass


class PauliInterface(TypedDict):
    """
    Definition of Pauli parameters
    must match the schema Input
    from orchestration API
    """
    operators: list
    qbits: int
    pauli_weight: int
    sigma_sq: NotRequired[dict[str, float]]
    sigma_sq_optimization_kwargs: NotRequired[dict[str, Any]]
    seed: NotRequired[int]
    target_operators: NotRequired[list]


class EnsembleInterface(TypedDict):
    """
    Definition of Ensemble parameters
    must match the schema Input
    from orchestration API
    """
    operators: list
    dim: int
    num_axes: int
    sigma_sq: NotRequired[dict[str, float]]
    sigma_sq_optimization_kwargs: NotRequired[dict[str, Any]]
    seed: NotRequired[int]
    target_operators: NotRequired[list]


Interface: TypeAlias = PauliInterface | EnsembleInterface | EMPTY_MODEL_PARAMS
TypeInterface: TypeAlias = Type[PauliInterface] | Type[EnsembleInterface]


VALID_MODEL_PARAMS: dict[Model, TypeInterface] = {
    Model.pauli: PauliInterface,
    Model.ensemble: EnsembleInterface,
}


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


class TrainingParameters(TypedDict):
    """
    Definition of train fct parameters
    must match the schema Input
    from orchestration API
    """
    batch_size: int
    num_passes: int
    weight_optimization_kwargs: WeightParams
    state_kwargs: StateParams


def encode_base64(data: pd.DataFrame) -> str:
    """
    take a normal pandas dataframe and encode as
    base64 "string" of csv export

    Parameters:
    -----------
    data: pd.DataFrame

    Returns:
    str
    """
    raw_string: str = data.to_csv(index=False)
    raw_bytes: bytes = raw_string.encode("ascii")
    base64_bytes = base64.b64encode(raw_bytes)
    base64_string = base64_bytes.decode("ascii")
    return base64_string


class DataSerializer:
    """
    Convenience wrapper class for uploading dataset
    """
    def __init__(
        self,
        model: ModelClient,
        data: pd.DataFrame,
    ):
        """
        Create a DataSerializer instance from a pandas DataFrame

        Parameters:
        -----------
        model: ModelClient
            the parent model client
        data: pd.DataFrame:
            the dataset as a DataFrame

        Returns:
        --------
        DataSerializer instance
        """
        self.model: ModelClient = model
        self.client: QcogClient = model.client
        self.data: pd.DataFrame = data

    def upload(self) -> dict:
        """
        upload the dataset via API call
        """
        self.dataset: dict = self.client.post("dataset", self.payload)
        return self.dataset

    @property
    def payload(self) -> dict:
        """
        encode dataframe as expected blob format:
        """
        return {
            "format": "csv",
            "source": "client",
            "data": encode_base64(self.data),
            "project_guid": self.client.project["guid"],
        }


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
        client: QcogClient | None = None,
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
        self.client: QcogClient = QcogClient() if client is None else client
        self.version: str = version
        self.dataset: dict = {}
        self.training_parameters: dict = {}
        self.trained_model: dict = {}

    def _preload(self, ep: str, guid: str) -> dict:
        """
        Utility function

        Parameters:
        -----------
        ep: str
            endpoint name (ex: dataset)
        guid: str
            endpoint name (ex: dataset)

        Returns:
        --------
        dict response from api call
        """
        return self.client.get(f"{ep}/{guid}")

    def _training_parameters(self, params: TrainingParameters) -> None:
        """
        Upload training parameters

        Parameters:
        -----------
        params: TrainingParameters
            Valid TypedDict of the training parameters
        """
        self.training_parameters = self.client.post(
            "training_parameters",
            {
                "project_guid": self.client.project["guid"],
                "model": self.model.value,
                "parameters": {
                    "model": self.params
                } | params
            }
        )

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
            raise ValueError(f"Invalid type of model parameters got {type(params)} expected {VALID_MODEL_PARAMS[self.model])}")  # noqa: 503
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

    def data(self, data: pd.DataFrame, upload: bool = True) -> ModelClient:
        """
        For a fresh "to train" model and properly initialized model
        upload a pandas DataFrame dataset.

        Parameters:
        -----------
        data: pd.DataFrame:
            the dataset as a DataFrame
        upload: bool:
            if true post the dataset

        Returns:
        --------
        ModelClient itself
        """
        self.data_serializer: DataSerializer = DataSerializer(self, data)
        if upload:
            return self.data_upload()
        return self

    def data_upload(self) -> ModelClient:
        """
        Use dataserializer to upload a dataset
        """
        self.dataset = self.data_serializer.upload()
        return self

    def preloaded_data(self, guid: str) -> ModelClient:
        """
        retrieve a dataset that was previously uploaded from guid.

        Parameters:
        -----------
        guid: str:
            guid of a previously uploaded dataset

        Returns:
        --------
        ModelClient itself
        """
        self.dataset = self._preload("dataset", guid)
        return self

    def preloaded_training_parameters(
        self, guid: str,
        rebuild: bool = False
    ) -> ModelClient:
        """
        Retrieve preexisting training parameters payload.

        Parameters:
        -----------
        guid: str
            model guid
        rebuild: bool
            if True, will initialize the class "model"
            (ex: pauli or ensemble) from the payload

        Returns:
        --------
        ModelClient itself
        """
        self.training_parameters = self._preload(
            "training_parameters",
            guid,
        )
        if rebuild:
            return self._rebuild_model_params()
        return self

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

    def train(self, params: TrainingParameters) -> ModelClient:
        """
        For a fresh "to train" model properly configured and initialized
        trigger a training request.

        Parameters:
        -----------
        params: TrainingParameters

        Returns:
        --------
        ModelClient itself
        """
        self._training_parameters(params)
        self.trained_model = self.client.post(
            "model",
            {
                "training_parameters_guid": self.training_parameters["guid"],
                "dataset_guid": self.dataset["guid"],
                "project_guid": self.client.project["guid"],
                "training_package_location": f"s3://ubiops-qognitive-default/packages/qcog-{self.version}-cp310-cp310-linux_x86_64/training_package.zip"  # noqa: 503
            },
        )
        return self

    def status(self) -> dict:
        return self.client.get(f"model/{self.trained_model['guid']}")

    def forecast(
        self,
        data: pd.DataFrame,
        parameters: dict,
    ) -> dict:  # pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
        """
        see inference
        """
        print(
                "\"forecast\" is deprecated, please use \"inference\" method instead"  # noqa: 503
        )
        return self.inference(data, parameters)

    def inference(
        self,
        data: pd.DataFrame,
        parameters: dict,
    ) -> dict:  # pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
        """
        From a trained model query an inference.

        Parameters:
        -----------
        data: pd.DataFrame:
            the dataset as a DataFrame
        parameters: dict:
            inference parameters

        Returns:
        --------
        dict: the API call response as dict json

        """
        resp: dict = self.client.post(
            f"model/{self.trained_model['guid']}/inference",
            {
                "data": encode_base64(data),
                "parameters": parameters
            },
        )
        return resp
