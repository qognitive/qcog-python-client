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
    pass


class PauliInterface(TypedDict):
    operators: list
    qbits: int
    pauli_weight: int
    sigma_sq: NotRequired[dict[str, float]]
    sigma_sq_optimization_kwargs: NotRequired[dict[str, Any]]
    seed: NotRequired[int]
    target_operators: NotRequired[list]


class EnsembleInterface(TypedDict):
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
    learning_rate: float
    iterations: int
    optimization_method: str
    step_size: float
    first_moment_decay: float
    second_moment_decay: float
    epsilon: float


class FisherParams(TypedDict):
    learning_rate: float


class StateParams(TypedDict):
    state_method: str
    iterations: int
    learning_rate_axes: float
    fisher_axes_kwargs: FisherParams
    fisher_state_kwargs: FisherParams


class TrainingParameters(TypedDict):
    batch_size: int
    num_passes: int
    weight_optimization_kwargs: WeightParams
    state_kwargs: StateParams


def encode_base64(data: pd.DataFrame) -> str:
    raw_string: str = data.to_csv(index=False)
    raw_bytes: bytes = raw_string.encode("ascii")
    base64_bytes = base64.b64encode(raw_bytes)
    base64_string = base64_bytes.decode("ascii")
    return base64_string


class DataSerializer:
    def __init__(
        self,
        model: ModelClient,
        data: pd.DataFrame,
    ):
        self.model: ModelClient = model
        self.client: QcogClient = model.client
        self.data: pd.DataFrame = data

    def upload(self) -> dict:
        self.dataset: dict = self.client.post("dataset", self.payload)
        return self.dataset

    @property
    def payload(self) -> dict:

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
        self.model: Model = Model("NOT_SET")
        self.params: Interface = EMPTY_MODEL_PARAMS()
        self.client: QcogClient = QcogClient() if client is None else client
        self.version: str = version
        self.dataset: dict = {}
        self.training_parameters: dict = {}
        self.trained_model: dict = {}

    def _preload(self, ep: str, guid: str) -> dict:
        return self.client.get(f"{ep}/{guid}")

    def _training_parameters(self, params: TrainingParameters) -> None:
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
        self._model(self.training_parameters["model"])
        copy_params: dict = self.training_parameters[
            "parameters"
        ]["model"].copy()
        model_params: Interface = VALID_MODEL_PARAMS[self.model](
            **copy_params
        )

        return self._params(model_params)

    def _model(self, model: str) -> ModelClient:
        self.model = Model(model)
        return self

    def _params(self, params: Interface) -> ModelClient:
        if not params:
            raise ValueError("Invalid empty model parameters")
        self.params = params
        return self

    def model_params(
        self,
        model: str,
        params: Interface,
    ) -> ModelClient:
        return self._model(model)._params(params)

    def data(self, data: pd.DataFrame, upload: bool = True) -> ModelClient:
        self.data_serializer: DataSerializer = DataSerializer(self, data)
        if upload:
            self.dataset = self.data_serializer.upload()
        return self

    def preloaded_data(self, guid: str) -> ModelClient:
        self.dataset = self._preload("dataset", guid)
        return self

    def preloaded_training_parameters(
        self, guid: str,
        rebuild: bool = False
    ) -> ModelClient:
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
        print(
                "\"forecast\" is deprecated, please use \"inference\" method instead"  # noqa: 503
        )
        return self.inference(data, parameters)

    def inference(
        self,
        data: pd.DataFrame,
        parameters: dict,
    ) -> dict:  # pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
        resp: dict = self.client.post(
            f"model/{self.trained_model['guid']}/inference",
            {
                "data": encode_base64(data),
                "parameters": parameters
            },
        )
        return resp
