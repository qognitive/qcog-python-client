"""Qcog API client."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Generic, Type, TypeAlias, TypeVar

import pandas as pd

from qcog_python_client.schema.common import Model
from qcog_python_client.schema.generated_schema.models import (
    ModelEnsembleParameters,
    ModelGeneralParameters,
    ModelPauliParameters,
    TrainingStatus,
)

from ._jsonable_parameters import (
    jsonable_inference_parameters,
    jsonable_train_parameters,
)
from .client import (
    AIOHTTPClient,
    RequestsClient,
    base642dataframe,
    encode_base64,
)
from .schema import (
    AsyncInferenceProtocol,
    AsyncTrainProtocol,
    DatasetPayload,
    InferenceParameters,
    InferenceProtocol,
    NotRequiredStateParams,
    NotRequiredWeightParams,
    Operator,
    TrainingParameters,
    TrainProtocol,
)

TrainingModel: TypeAlias = (
    ModelPauliParameters | ModelEnsembleParameters | ModelGeneralParameters
)

CLIENT = TypeVar("CLIENT")


ModelMap: dict[Model, Type[TrainingModel]] = {
    Model.pauli: ModelPauliParameters,
    Model.ensemble: ModelEnsembleParameters,
    Model.general: ModelGeneralParameters,
}

# TODO: Move to the schema as Enum
# https://ubiops.com/docs/r_client_library/deployment_requests/#response-structure_1
# WAITING_STATUS = ("processing", "pending")
WAITING_STATUS = (TrainingStatus.processing, TrainingStatus.pending)
SUCCESS_STATUS = (TrainingStatus.completed,)


DEFAULT_QCOG_VERSION = "0.0.70"


def numeric_version(version: str) -> list[int]:
    """Reformulate a string M.N.F version for test comparison.

    Parameters
    ----------
    version : str
        expected to be of the form M.N.F

    Return
    ------
    list[int]
        a list of 3 int that can pythonically compared

    """
    numbers = version.split(".")
    if len(numbers) != 3:
        raise ValueError(f"Invalid version number {version}")

    return [int(w) for w in numbers]


class StateNotSetError(Exception):
    """Exception raised when a state is not set."""

    def __init__(self, missing_state: str, suggestion: str | None = None) -> None:
        """Initialize the exception.

        Attributes
        ----------
        missing_state : str
            The state that is missing.

        suggestion : str | None
            A suggestion for remediation.

        """
        self.missing_state = missing_state
        self.suggestion = suggestion
        super().__init__(f"Missing state: {missing_state}. {suggestion or ''}")


class BaseQcogClient(Generic[CLIENT]):  # noqa: D101
    def __init__(self) -> None:  # noqa: D107
        self._version: str
        self._http_client: CLIENT
        self._model: TrainingModel | None = None
        self._project: dict | None = None
        self._dataset: dict | None = None
        self._training_parameters: dict | None = None
        self._trained_model: dict | None = None
        self._inference_result: dict | None = None

    @property
    def model(self) -> TrainingModel:
        """Return the model."""
        if self._model is None:
            raise StateNotSetError(
                "model", "Please set the model first using `pauli` or `ensemble` method"
            )
        return self._model

    @property
    def project(self) -> dict:
        """Return the project."""
        if self._project is None:
            raise StateNotSetError(
                "No project has been found associated with this request."
            )
        return self._project

    @property
    def dataset(self) -> dict:
        """Return the dataset."""
        if self._dataset is None:
            raise StateNotSetError(
                "No dataset has been found associated with this request.",
                """You can use `data` method to upload a new dataset or
                `preloaded_data` to load an existing one""",
            )
        return self._dataset

    @property
    def training_parameters(self) -> dict:
        """Return the training parameters."""
        if self._training_parameters is None:
            raise StateNotSetError(
                "No training parameters have been found associated with this request.",
                """You can use `train` method to start a new training or
                `preloaded_training_parameters` to load existing ones""",  # noqa: 501
            )
        return self._training_parameters

    @property
    def trained_model(self) -> dict:
        """Return the trained model."""
        if self._trained_model is None:
            raise StateNotSetError(
                "No trained model has been found associated with this request.",
                """You can use `train` method to start a new training or
                `preloaded_model` to load an existing one""",
            )
        return self._trained_model

    @property
    def inference_result(self) -> dict:
        """Return the inference result."""
        if self._inference_result is None:
            raise StateNotSetError(
                "No inference result has been found associated with this request.",
                "You can use `inference` method to run an inference",
            )
        return self._inference_result

    @property
    def version(self) -> str:
        """Qcog version."""
        return self._version

    @version.setter
    def version(self, value: str) -> None:
        numeric_version(value)  # validate version format
        self._version = value

    @property
    def http_client(self) -> CLIENT:
        """Return the http client."""
        return self._http_client

    @http_client.setter
    def http_client(self, value: CLIENT) -> None:
        self._http_client = value

    def pauli(
        self,
        operators: list[Operator],
        qbits: int = 2,
        pauli_weight: int = 2,
        sigma_sq: dict[str, float] = {},
        sigma_sq_optimization: dict[str, float] = {},
        seed: int = 42,
        target_operator: list[Operator] = [],
    ) -> Any:
        """Select PauliModel for the training."""
        self._model = ModelPauliParameters(
            operators=[str(op) for op in operators],
            qbits=qbits,
            pauli_weight=pauli_weight,
            sigma_sq=sigma_sq,
            sigma_sq_optimization_kwargs=sigma_sq_optimization,
            seed=seed,
            target_operators=[str(op) for op in target_operator],
            model_name=Model.pauli.value,
        )
        return self

    def ensemble(
        self,
        operators: list[Operator],
        dim: int = 16,
        num_axes: int = 4,
        sigma_sq: dict[str, float] = {},
        sigma_sq_optimization: dict[str, float] = {},
        seed: int = 42,
        target_operator: list[Operator] = [],
    ) -> Any:
        """Select EnsembleModel for the training."""
        # Cast all the operators to string
        self._model = ModelEnsembleParameters(
            operators=[str(op) for op in operators],
            dim=dim,
            num_axes=num_axes,
            sigma_sq=sigma_sq,
            sigma_sq_optimization_kwargs=sigma_sq_optimization,
            seed=seed,
            target_operators=[str(op) for op in target_operator],
            model_name=Model.ensemble.value,
        )
        return self


class QcogClient(  # noqa: D101
    BaseQcogClient[RequestsClient],
    TrainProtocol,
    InferenceProtocol,
):
    @classmethod
    def create(
        cls,
        *,
        token: str | None = None,
        hostname: str = "dev.qognitive.io",
        port: int = 443,
        api_version: str = "v1",
        safe_mode: bool = False,
        version: str = DEFAULT_QCOG_VERSION,
    ) -> QcogClient:
        """Create a client with initialization from the API.

        Since __init__ is always sync we cannot call to the API using that
        method of class creation. If we need to fetch things such as the
        project ID that the token is associated with the only way to do that
        properly with async objects is to use a factory method.

        Here we replace init with create and then once the object is created
        (since the sync part is linked to the creation of the object in memory
        space) we are able to then call the API using our async methods and
        not block on IO.

        Qcog api client implementation there are 2 main expected usages:

        1. Training
        2. Inference

        The class definition is such that every parameter must be used
        explicitly:

        .. code-block:: python

            hsm_client = QcogClient(token="value", version="0.0.45")

        Each "public" method return "self" to chain method calls unless
        it is one of the following utilities: status and inference

        Each method that results in an api call will store the api
        response as a json dict in a class attribute

        In practice, the 2 main expected usage would be for a fresh training:

        .. code-block:: python

            hsm = QcogClient.create(...).pauli(...).data(...).train(...)

        where the "..." would be replaced with desired parametrization

        If we wanted, we could infer after training, right away.

        .. code-block:: python

            result: pd.DataFrame = hsm.inference(...)

        but this would require to run the following loop:

        .. code-block:: python

            hsm.wait_for_training().inference(...)

        to make sure training has successfully completed.

        To run multiple inference on a persistent trained model,
        the trained_model guid go to storage. Datasets? Also
        storage. Training parameters? Storage. That way one can
        rebuild the client to run inference:

        .. code-block:: python

            hsm = QcogClient.create(...).preloaded_model(trained_model_guid)

            for df in list_of_dataframes:
                result: Dataframe = hsm.inference(...)

        Most methods class order is not important with 3 exceptions:

        1. train may only be called after data, and named model
        2. inference and status must have a preloaded model first

        Parameters
        ----------
        token : str | None
            A valid API token granting access optional
            when unset (or None) expects to find the proper
            value as QCOG_API_TOKEN environment variable
        hostname : str
            API endpoint hostname, currently defaults to dev.qognitive.io
        port : int
            port value, default to https 443
        api_version : str
            the "vX" part of the url for the api version
        safe_mode : bool
            if true runs healthchecks before running any api call
            sequences
        version : str
            the qcog version to use. Must be no smaller than `OLDEST_VERSION`
            and no greater than `NEWEST_VERSION`

        Returns
        -------
        QcogClient
            the client object

        """
        hsm = cls()
        hsm.version = version
        hsm.http_client = RequestsClient(
            token=token,
            hostname=hostname,
            port=port,
            api_version=api_version,
        )

        if safe_mode:
            hsm.http_client.get("status")

        # hsm._project = hsm.http_client.get("bootstrap")

        return hsm

    def _preload(self, ep: str, guid: str) -> dict:
        """Pre Load Utility function.

        Parameters
        ----------
        ep : str
            endpoint name (ex: dataset)
        guid : str
            endpoint name (ex: dataset)

        Returns
        -------
        dict
            response from api call

        """
        return self.http_client.get(f"{ep}/{guid}")

    def _preload_training_parameters(self, params: TrainingParameters) -> None:
        """Upload training parameters.

        Parameters
        ----------
        params : TrainingParameters
            Valid TypedDict of the training parameters

        """
        self._training_parameters = self.http_client.post(
            "training_parameters",
            {
                # "project_guid": self.project["guid"],
                "model": self.model.model_name,
                "parameters": {"model": self.model.model_dump()}
                | jsonable_train_parameters(params),
            },
        )

    def data(self, data: pd.DataFrame) -> QcogClient:
        """Upload a dataset for training.

        For a fresh "to train" model and properly initialized model
        upload a pandas DataFrame dataset.

        Parameters
        ----------
        data : pd.DataFrame:
            the dataset as a DataFrame
        upload : bool:
            if true post the dataset

        Returns
        -------
        AsyncQcogClient

        """
        # Validate data payload
        data_payload = DatasetPayload(
            format="dataframe",
            source="client",
            data=encode_base64(data),
            project_guid=None,
        ).model_dump()

        valid_data: dict = {k: v for k, v in data_payload.items()}  # type cast
        self._dataset = self.http_client.post("dataset", valid_data)
        return self

    def preloaded_data(self, guid: str) -> QcogClient:
        """Retrieve a dataset that was previously uploaded from guid.

        Parameters
        ----------
        guid : str
            guid of a previously uploaded dataset

        Returns
        -------
        QcogClient
            itself

        """
        self._dataset = self._preload("dataset", guid)
        return self

    def preloaded_training_parameters(
        self, guid: str, rebuild: bool = False
    ) -> QcogClient:
        """Retrieve preexisting training parameters payload.

        Parameters
        ----------
        guid : str
            model guid
        rebuild : bool
            if True, will initialize the class "model"
            (ex: pauli or ensemble) from the payload

        Returns
        -------
        QcogClient
            itself

        """
        self._training_parameters = self._preload(
            "training_parameters",
            guid,
        )
        return self

    def preloaded_model(self, guid: str) -> QcogClient:
        """Preload a model from a guid."""
        self._trained_model = self._preload("model", guid)

        self.version = self.trained_model["qcog_version"]

        self.preloaded_training_parameters(
            self.trained_model["training_parameters_guid"]
        )

        # The model name in order to retrieve the correct parameters
        model_name = Model(self.training_parameters["model"])

        # Parameters for the model
        model_parameters = self.training_parameters["parameters"]["model"]
        # Retrieve the validation for the model parameters
        validate_cls = ModelMap[model_name]
        # Validate the model parameters
        self._model = validate_cls(**model_parameters)
        return self

    def train(
        self,
        batch_size: int,
        num_passes: int,
        weight_optimization: NotRequiredWeightParams,
        get_states_extra: NotRequiredStateParams,
    ) -> QcogClient:
        """Start a training job.

        For a fresh "to train" model properly configured and initialized trigger
        a training request.

        Parameters
        ----------
        batch_size : int
            The number of samples to use in each training batch.
        num_passes : int
            The number of passes through the dataset.
        weight_optimization : NotRequiredWeightParams
            optimization parameters for the weights
        get_states_extra : NotRequiredStateParams
            optimization parameters for the states

        Returns
        -------
        AsyncQcogClient

        """
        params: TrainingParameters = TrainingParameters(
            batch_size=batch_size,
            num_passes=num_passes,
            weight_optimization_kwargs=weight_optimization,
            state_kwargs=get_states_extra,
        )

        self._preload_training_parameters(params)

        self._trained_model = self.http_client.post(
            "model",
            {
                "training_parameters_guid": self.training_parameters["guid"],
                "dataset_guid": self.dataset["guid"],
                "qcog_version": self.version,
            },
        )

        return self

    def status(self) -> TrainingStatus:
        """Fetch the status of the training request."""
        self.status_resp: dict = self.http_client.get(
            f"model/{self.trained_model['guid']}"
        )

        self.last_status = TrainingStatus(self.status_resp["status"])
        return self.last_status

    def wait_for_training(self, poll_time: int = 60) -> QcogClient:
        """Wait for training to complete.

        Note
        ----
        This function is blocking

        Parameters
        ----------
        poll_time : int:
            status checks intervals in seconds

        Returns
        -------
        QcogClient
            itself

        """
        while self.status() in WAITING_STATUS:
            if self.last_status in SUCCESS_STATUS:
                break
            time.sleep(poll_time)

        if self.last_status not in SUCCESS_STATUS:
            # something went wrong
            raise RuntimeError(
                f"something went wrong {json.dumps(self.status_resp, indent=4)}"  # noqa: 503
            )

        return self

    def inference(
        self,
        data: pd.DataFrame,
        parameters: InferenceParameters,
    ) -> pd.DataFrame:
        """From a trained model query an inference.

        Parameters
        ----------
        data : pd.DataFrame
            the dataset as a DataFrame
        parameters : dict
            inference parameters

        Returns
        -------
        pd.DataFrame
            the predictions

        """
        inference_result = self.http_client.post(
            f"model/{self.trained_model['guid']}/inference",
            {
                "data": encode_base64(data),
                "parameters": jsonable_inference_parameters(parameters),
            },
        )

        print(inference_result)

        return base642dataframe(
            inference_result["response"]["data"],
        )

    def pauli(  # noqa: D102
        self,
        operators: list[Operator],
        qbits: int = 2,
        pauli_weight: int = 2,
        sigma_sq: dict[str, float] = {},
        sigma_sq_optimization: dict[str, float] = {},
        seed: int = 42,
        target_operator: list[Operator] = [],
    ) -> QcogClient:
        self = super().pauli(
            operators,
            qbits,
            pauli_weight,
            sigma_sq,
            sigma_sq_optimization,
            seed,
            target_operator,
        )

        return self

    def ensemble(  # noqa: D102
        self,
        operators: list[Operator],
        dim: int = 16,
        num_axes: int = 4,
        sigma_sq: dict[str, float] = {},
        sigma_sq_optimization: dict[str, float] = {},
        seed: int = 42,
        target_operator: list[Operator] = [],
    ) -> QcogClient:
        self = super().ensemble(
            operators,
            dim,
            num_axes,
            sigma_sq,
            sigma_sq_optimization,
            seed,
            target_operator,
        )

        return self


class AsyncQcogClient(  # noqa: D101
    BaseQcogClient[AIOHTTPClient],
    AsyncTrainProtocol,
    AsyncInferenceProtocol,
):
    @classmethod
    async def create(
        cls,
        *,
        token: str | None = None,
        hostname: str = "dev.qognitive.io",
        port: int = 443,
        api_version: str = "v1",
        safe_mode: bool = False,
        version: str = DEFAULT_QCOG_VERSION,
    ) -> AsyncQcogClient:
        """Asyncronous Qcog api client implementation.

        This is similar to the sync client with the exception that any API
        calls will be async and require an await

        For example:

        .. code-block:: python

            hsm = (await AsyncQcogClient.create(...)).pauli(...)
            await hsm.data(...)
            await hsm.train(...)

        where the "..." would be replaced with desired parametrization

        If we wanted, we could infer after training, right away.

        .. code-block:: python

            result: pd.DataFrame = await hsm.inference(...)

        but this would require us to explicitly wait for training to complete

        .. code-block:: python

            await hsm.wait_for_training()
            result: pd.DataFrame = await hsm.inference(...)

        to make sure training has successfully completed.

        Parameters
        ----------
        token : str | None
            A valid API token granting access optional
            when unset (or None) expects to find the proper
            value as QCOG_API_TOKEN environment variable
        hostname : str
            API endpoint hostname, currently defaults to dev.qognitive.io
        port : int
            port value default to https 443
        api_version : str
            the "vX" part of the url for the api version
        safe_mode : bool
            if true runs healthchecks before running any api call
            sequences
        version : str
            the qcog version to use. Must be no smaller than `OLDEST_VERSION`
            and no greater than `NEWEST_VERSION`

        Returns
        -------
        AsyncQcogClient
            the client object

        """
        hsm = cls()
        hsm.version = version
        hsm.http_client = AIOHTTPClient(
            token=token,
            hostname=hostname,
            port=port,
            api_version=api_version,
        )

        if safe_mode:
            await hsm.http_client.get("status")

        return hsm

    async def _preload(self, ep: str, guid: str) -> dict:
        """Pre load Utility function.

        Parameters
        ----------
        ep : str
            endpoint name (ex: dataset)
        guid : str
            endpoint name (ex: dataset)

        Returns
        -------
        dict
            response from api call

        """
        return await self.http_client.get(f"{ep}/{guid}")

    async def _preload_training_parameters(self, params: TrainingParameters) -> None:
        """Upload training parameters.

        Parameters
        ----------
        params : TrainingParameters
            Valid TypedDict of the training parameters

        """
        self._training_parameters = await self.http_client.post(
            "training_parameters",
            {
                # "project_guid": self.project["guid"],
                "model": self.model.model_name,
                "parameters": {"model": self.model.model_dump()}
                | jsonable_train_parameters(params),
            },
        )

    async def data(self, data: pd.DataFrame) -> AsyncQcogClient:
        """Upload a dataset for training.

        For a fresh "to train" model and properly initialized model
        upload a pandas DataFrame dataset.

        Parameters
        ----------
        data : pd.DataFrame:
            the dataset as a DataFrame
        upload : bool:
            if true post the dataset

        Returns
        -------
        AsyncQcogClient
            itself

        """
        # Validate data payload
        data_payload = DatasetPayload(
            format="dataframe",
            source="client",
            data=encode_base64(data),
            project_guid=None,
        ).model_dump()

        valid_data: dict = {k: v for k, v in data_payload.items()}
        self._dataset = await self.http_client.post("dataset", valid_data)
        return self

    async def preloaded_data(self, guid: str) -> AsyncQcogClient:
        """Retrieve a dataset that was previously uploaded from guid.

        Parameters
        ----------
        guid : str
            guid of a previously uploaded dataset

        Returns
        -------
        AsyncQcogClient
            itself

        """
        self._dataset = await self._preload("dataset", guid)
        return self

    async def preloaded_training_parameters(
        self, guid: str, rebuild: bool = False
    ) -> AsyncQcogClient:
        """Retrieve preexisting training parameters payload.

        Parameters
        ----------
        guid: str
            model guid
        rebuild: bool
            if True, will initialize the class "model"
            (ex: pauli or ensemble) from the payload

        Returns
        -------
        AsyncQcogClient
            itself

        """
        self._training_parameters = await self._preload(
            "training_parameters",
            guid,
        )
        return self

    async def preloaded_model(self, guid: str) -> AsyncQcogClient:
        """Preload a model from a guid."""
        self._trained_model = await self._preload("model", guid)

        self.version = self.trained_model["qcog_version"]

        await self.preloaded_training_parameters(
            self.trained_model["training_parameters_guid"]
        )

        # The model name in order to retrieve the correct parameters
        model_name = Model(self.training_parameters["model"])

        # Parameters for the model
        model_parameters = self.training_parameters["parameters"]["model"]
        # Retrieve the validation for the model parameters
        validate_cls = ModelMap[model_name]
        # Validate the model parameters
        self._model = validate_cls(**model_parameters)
        return self

    async def train(
        self,
        batch_size: int,
        num_passes: int,
        weight_optimization: NotRequiredWeightParams,
        get_states_extra: NotRequiredStateParams,
    ) -> AsyncQcogClient:
        """Start a training job.

        For a fresh "to train" model properly configured and initialized trigger
        a training request.

        Parameters
        ----------
        batch_size : int
            The number of samples to use in each training batch.
        num_passes : int
            The number of passes through the dataset.
        weight_optimization : NotRequiredWeightParams
            optimization parameters for the weights
        get_states_extra : NotRequiredStateParams
            optimization parameters for the states

        Returns
        -------
        AsyncQcogClient

        """
        params: TrainingParameters = TrainingParameters(
            batch_size=batch_size,
            num_passes=num_passes,
            weight_optimization_kwargs=weight_optimization,
            state_kwargs=get_states_extra,
        )

        await self._preload_training_parameters(params)

        self._trained_model = await self.http_client.post(
            "model",
            {
                "training_parameters_guid": self.training_parameters["guid"],
                "dataset_guid": self.dataset["guid"],
                "qcog_version": self.version,
            },
        )

        return self

    async def status(self) -> str:
        """Fetch the status of the training request."""
        self.status_resp: dict = await self.http_client.get(
            f"model/{self.trained_model['guid']}"
        )

        self.last_status: str = self.status_resp["status"]
        return self.last_status

    async def wait_for_training(self, poll_time: int = 60) -> AsyncQcogClient:
        """Wait for training to complete.

        Parameters
        ----------
        poll_time : int
            status checks intervals in seconds

        Returns
        -------
        AsyncQcogClient
            itself

        """
        while (await self.status()) in WAITING_STATUS:
            if self.last_status in SUCCESS_STATUS:
                break
            await asyncio.sleep(poll_time)

        if self.last_status not in SUCCESS_STATUS:
            # something went wrong
            raise RuntimeError(
                f"something went wrong {json.dumps(self.status_resp, indent=4)}"  # noqa: 503
            )

        return self

    async def inference(
        self,
        data: pd.DataFrame,
        parameters: InferenceParameters,
    ) -> pd.DataFrame:
        """From a trained model query an inference.

        Parameters
        ----------
        data : pd.DataFrame
            the dataset as a DataFrame
        parameters : dict
            inference parameters

        Returns
        -------
        pd.DataFrame
            the predictions

        """
        self._inference_result: dict = await self.http_client.post(
            f"model/{self.trained_model['guid']}/inference",
            {
                "data": encode_base64(data),
                "parameters": jsonable_inference_parameters(parameters),
            },
        )

        return base642dataframe(
            self.inference_result["response"]["data"],
        )

    def pauli(  # noqa: D102
        self,
        operators: list[Operator],
        qbits: int = 2,
        pauli_weight: int = 2,
        sigma_sq: dict[str, float] = {},
        sigma_sq_optimization: dict[str, float] = {},
        seed: int = 42,
        target_operator: list[Operator] = [],
    ) -> AsyncQcogClient:
        self = super().pauli(
            operators,
            qbits,
            pauli_weight,
            sigma_sq,
            sigma_sq_optimization,
            seed,
            target_operator,
        )

        return self

    def ensemble(  # noqa: D102
        self,
        operators: list[Operator],
        dim: int = 16,
        num_axes: int = 4,
        sigma_sq: dict[str, float] = {},
        sigma_sq_optimization: dict[str, float] = {},
        seed: int = 42,
        target_operator: list[Operator] = [],
    ) -> AsyncQcogClient:
        self = super().ensemble(
            operators,
            dim,
            num_axes,
            sigma_sq,
            sigma_sq_optimization,
            seed,
            target_operator,
        )

        return self
