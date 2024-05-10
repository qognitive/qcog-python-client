from __future__ import annotations

import asyncio
from typing import Type, TypeAlias
import json
import time
import pandas as pd

from .client import (
    AIOHTTPClient,
    RequestsClient,
    base642dataframe,
    encode_base64,
)
from .schema import (
    Model,
    Dataset,
    TrainProtocol,
    TrainingParameters,
    InferenceProtocol,
    InferenceParameters,
    Operator,
    NotRequiredWeightParams,
    NotRequiredStateParams,
    PauliModel,
    EnsembleModel,
)


TrainingModel: TypeAlias = PauliModel | EnsembleModel


MODEL_MAP: dict[str, Type[TrainingModel]] = {
    Model.pauli.value: PauliModel,
    Model.ensemble.value: EnsembleModel,
}


def numeric_version(version: str) -> list[int]:
    """
    Reformulate a string M.N.F version for test comparison

    Parameters:
    -----------
    version: str
        expected to be of the form M.N.F

    Return:
    -------
    list[int]: a list of 3 int that can pythonically compared
    """
    numbers = version.split(".")
    if len(numbers) != 3:
        raise ValueError(f"Invalid version number {version}")

    return [int(w) for w in numbers]


class BaseQcogClient(TrainProtocol, InferenceProtocol):

    OLDEST_VERSION = "0.0.43"
    NEWEST_VERSION = "0.0.44"
    PROJECT_GUID_TEMPORARY: str = "45ec9045-3d50-46fb-a82c-4aa0502801e9"

    def __init__(self, *, version: str = NEWEST_VERSION):
        self.model: PauliModel | EnsembleModel

        if numeric_version(version) < numeric_version(self.OLDEST_VERSION):
            raise ValueError(
                f"qcog version can't be older than {self.OLDEST_VERSION}"
            )
        if numeric_version(version) > numeric_version(self.NEWEST_VERSION):
            raise ValueError(
                f"qcog version can't be older than {self.NEWEST_VERSION}"
            )
        self.version: str = version
        self.project: dict[str, str]
        self.dataset: dict = {}
        self.training_parameters: dict = {}
        self.trained_model: dict = {}
        self.inference_result: dict = {}

    def pauli(
        self,
        operators: list[Operator],
        qbits: int = 2,
        pauli_weight: int = 2,
        sigma_sq: dict[str, float] = {},
        sigma_sq_optimization: dict[str, float] = {},
        seed: int = 42,
        target_operator: list[Operator] = [],
    ) -> BaseQcogClient:
        """
        Select PauliModel for the training
        """
        self.model = PauliModel(
            operators,
            qbits,
            pauli_weight,
            sigma_sq,
            sigma_sq_optimization,
            seed,
            target_operator
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
    ) -> BaseQcogClient:
        """
        Select EnsembleModel for the training
        """
        self.model = EnsembleModel(
            operators,
            dim,
            num_axes,
            sigma_sq,
            sigma_sq_optimization,
            seed,
            target_operator
        )
        return self


class QcogClient(BaseQcogClient):

    def __init__(
        self,
        *,
        token: str | None = None,
        hostname: str | None = None,
        port: str | int | None = None,
        api_version: str = "v1",
        secure: bool = True,
        safe_mode: bool = True,  # NOTE will make False default later
        verify: bool = True,  # for debugging until ssl is fixed
        test_project: bool = False,
        version: str = BaseQcogClient.NEWEST_VERSION,
    ):
        """
        Qcog api client implementation there are 2 main expected usages:
            1. Training
            2. Inference

        The class definition is such that every parameter must be used
        explicitly:

            hsm_client = QcogClient(token="value", version="0.0.45")

        Each "public" method return "self" to chain method calls unless
        it is one of the following utilities: status and inference

        Each method that results in an api call will store the api
        response as a json dict in a class attribute

        In practice, the 2 main expected usage would be for a fresh training:

        hsm = QcogClient(...).pauli(...).data(...).train(...)

        where the "..." would be replaced with desired parametrization

        If we wanted, we could infer after training, right away.

        result: pd.DataFrame = hsm.inference(...)

        but this would require to run the following loop:

            hsm.wait_for_training().inference(...)

        to make sure training has successfully completed.

        To run multiple inference on a persistent trained model,
        the trained_model guid go to storage. Datasets? Also
        storage. Training parameters? Storage. That way one can
        rebuild the client to run inference:

        hsm = QcogClient(...).preloaded_model(trained_model_guid)

        for df in list_of_dataframes:
            result: Dataframe = hsm.inference(...)

        Most methods class order is not important with 3 exceptions:
            1. train may only be called after data, and named model
            2. inference and status must have a preloaded model first

        Parameters:
        -----------
        token: str | None
            A valid API token granting access optional
            when unset (or None) expects to find the proper
            value as QCOG_API_TOKEN environment veriable
        hostname: str | None
            optional string of the hostname. Currently default
            to a standard api endpoint
        port: str | int | None
            port value default to https 443
        api_version: str
            the "vX" part of the url for the api version
        secure: bool
            if true use https else use http mainly for local
            testing
        safe_mode: bool
            if true runs healthchecks before running any api call
            sequences
        verify: bool
            ignore ssl provenance for testing purposes
        test_projest: bool
            For testing purposes. if the project resolvers finds
            no project, create one. For testing purposes
        version: str
            the qcog version to use. Must be no smaller than OLDEST_VERSION
            and no greater than NEWEST_VERSION.
        """
        super().__init__(version=version)
        self.http_client = RequestsClient(
            token=token,
            hostname=hostname,
            port=port,
            api_version=api_version,
            secure=secure,
            safe_mode=safe_mode,
            verify=verify,
        )
        self._resolve_project(test_project)

    @classmethod
    def create(
        cls,
        *,
        token: str | None = None,
        hostname: str | None = None,
        port: str | int | None = None,
        api_version: str = "v1",
        secure: bool = True,
        safe_mode: bool = True,  # NOTE will make False default later
        verify: bool = True,  # for debugging until ssl is fixed
        test_project: bool = False,
        version: str = BaseQcogClient.NEWEST_VERSION,
    ) -> QcogClient:
        """Wrapper to maintain the same structure as the Async client."""
        return cls(
            token=token,
            hostname=hostname,
            port=port,
            api_version=api_version,
            secure=secure,
            safe_mode=safe_mode,
            verify=verify,
            test_project=test_project,
            version=version,
        )

    def _resolve_project(self, test_project: bool) -> None:
        """
        NOTE: CURRENTLY A STUB
        This method is a utility method of the class __init__
        method that resolves the project(s) accessible for this
        "token-as-proxy-for-org/user"

        Implements a "test mode" that will create a project

        Parameters:
        -----------
        test_project: bool
            If true, the class creation will create a new project and
            store its GUID in the PROJECT_GUID_TEMPORARY variable
        """

        if test_project:
            self.PROJECT_GUID_TEMPORARY = self.http_client.post(
                "project",
                {
                    "name": "poc-train-simple-model",
                    "bucket_name": "ubiops-qognitive-default"
                }
            )["guid"]
        self.project = self._preload("project", self.PROJECT_GUID_TEMPORARY)

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
        return self.http_client.get(f"{ep}/{guid}")

    def _training_parameters(self, params: TrainingParameters) -> None:
        """
        Upload training parameters

        Parameters:
        -----------
        params: TrainingParameters
            Valid TypedDict of the training parameters
        """
        self.training_parameters = self.http_client.post(
            "training_parameters",
            {
                "project_guid": self.project["guid"],
                "model": self.model.value,
                "parameters": {
                    "model": self.model.params
                } | params
            }
        )

    def data(
        self,
        data: pd.DataFrame
    ) -> QcogClient:
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
        QcogClient: itself
        """
        data_payload = Dataset(
            format="csv",
            source="client",
            data=encode_base64(data),
            project_guid=self.project["guid"],
        )
        valid_data: dict = {k: v for k, v in data_payload.items()}  # type cast
        self.dataset = self.http_client.post("dataset", valid_data)
        return self

    def preloaded_data(self, guid: str) -> QcogClient:
        """
        retrieve a dataset that was previously uploaded from guid.

        Parameters:
        -----------
        guid: str:
            guid of a previously uploaded dataset

        Returns:
        --------
        QcogClient itself
        """
        self.dataset = self._preload("dataset", guid)
        return self

    def preloaded_training_parameters(
        self, guid: str,
        rebuild: bool = False
    ) -> QcogClient:
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
        QcogClient itself
        """
        self.training_parameters = self._preload(
            "training_parameters",
            guid,
        )
        return self

    def preloaded_model(self, guid: str) -> QcogClient:
        self.trained_model = self._preload("model", guid)

        self._preload("project", self.trained_model["project_guid"])
        self.version = self.trained_model[
            "training_package_location"
        ].split("packages/")[-1].split("-")[1]

        self.preloaded_training_parameters(
            self.trained_model["training_parameters_guid"]
        )

        model_params = {
            k.replace(
                "_kwargs", ""
            ): v for k, v in self.training_parameters[
                "parameters"
            ][
                "model"
            ].items()
            if k != "model"
        }

        self.model = MODEL_MAP[
            self.training_parameters["model"]
        ](**model_params)
        return self

    def train(
        self,
        batch_size: int,
        num_passes: int,
        weight_optimization: NotRequiredWeightParams,
        get_states_extra: NotRequiredStateParams,
    ) -> QcogClient:
        """
        For a fresh "to train" model properly configured and initialized
        trigger a training request.

        Parameters:
        -----------
        params: TrainingParameters

        Returns:
        --------
        QcogClient: itself
        """

        params: TrainingParameters = TrainingParameters(
            batch_size=batch_size,
            num_passes=num_passes,
            weight_optimization_kwargs=weight_optimization,
            state_kwargs=get_states_extra,
        )

        self._training_parameters(params)

        self.trained_model = self.http_client.post(
            "model",
            {
                "training_parameters_guid": self.training_parameters["guid"],
                "dataset_guid": self.dataset["guid"],
                "project_guid": self.project["guid"],

                # TODO: we need to rewrite this to no expose internal details like that  # noqa: 503
                "training_package_location": f"s3://ubiops-qognitive-default/packages/qcog-{self.version}-cp310-cp310-linux_x86_64/training_package.zip"  # noqa: 503
            },
        )

        return self

    def status(self) -> str:
        self.status_resp: dict = self.http_client.get(
            f"model/{self.trained_model['guid']}"
        )

        self.last_status: str = self.status_resp["status"]
        return self.last_status

    def wait_for_training(self, poll_time: int = 5) -> QcogClient:
        """
        Wait for training to complete.

        the function is blocking

        Parameters:
        -----------
        poll_time: int:
            status checks intervals in seconds

        Returns:
        --------
        QcogClient: itself

        """
        while self.status() == "pending":
            if self.last_status == "completed":
                break
            time.sleep(poll_time)

        if self.last_status != "completed":
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
        pd.DataFrame: the predictions

        """
        self.inference_result = self.http_client.post(
            f"model/{self.trained_model['guid']}/inference",
            {
                "data": encode_base64(data),
                "parameters": parameters
            },
        )

        return base642dataframe(self.inference_result["response"]["data"])


class AsyncQcogClient(BaseQcogClient):

    def __init__(
        self,
        *,
        token: str | None = None,
        hostname: str | None = None,
        port: str | int | None = None,
        api_version: str = "v1",
        secure: bool = True,
        safe_mode: bool = True,  # NOTE will make False default later
        verify: bool = True,  # for debugging until ssl is fixed
        test_project: bool = False,
        version: str = BaseQcogClient.NEWEST_VERSION,
    ):
        """Asyncronous Qcog api client implementation

        This is similar to the sync client with the exception that any API
        calls will be async and require an await

        For example:
            hsm = await(
                (
                    await AsyncQcogClient(...).pauli(...).data(...)
                ).train(...)
            )

        where the "..." would be replaced with desired parametrization

        If we wanted, we could infer after training, right away.

        result: pd.DataFrame = await hsm.inference(...)

        but this would require us to explicitly wait for training to complete

            await hsm.wait_for_training()
            result: pd.DataFrame = await hsm.inference(...)

        to make sure training has successfully completed.

        Parameters:
        -----------
        See QcogClient for parameter details
        """
        super().__init__(version=version)
        self.http_client = AIOHTTPClient(
            token=token,
            hostname=hostname,
            port=port,
            api_version=api_version,
            secure=secure,
            safe_mode=safe_mode,
            verify=verify,
        )

    @classmethod
    async def create(
        cls,
        *,
        token: str | None = None,
        hostname: str | None = None,
        port: str | int | None = None,
        api_version: str = "v1",
        secure: bool = True,
        safe_mode: bool = True,  # NOTE will make False default later
        verify: bool = True,  # for debugging until ssl is fixed
        test_project: bool = False,
        version: str = BaseQcogClient.NEWEST_VERSION,
    ) -> AsyncQcogClient:
        """Factory method to create a client with intializations from the API.

        Since __init__ is always sync we cannot call to the API using that
        method of class creation. If we need to fetch things such as the
        project ID that the token is associated with the only way to do that
        properly with async objects is to use a factory method.

        Here we wrap init and then once the object is created (since the sync
        part is linked to the creation of the object in memory space) we are
        able to then call the API using our async methods and not block on IO.
        """
        hsm = cls(
            token=token,
            hostname=hostname,
            port=port,
            api_version=api_version,
            secure=secure,
            safe_mode=safe_mode,
            verify=verify,
            test_project=test_project,
            version=version,
        )
        await hsm._resolve_project(test_project)
        return hsm

    async def _resolve_project(self, test_project: bool) -> None:
        """
        NOTE: CURRENTLY A STUB
        This method is a utility method of the class __init__
        method that resolves the project(s) accessible for this
        "token-as-proxy-for-org/user"

        Implements a "test mode" that will create a project

        Parameters:
        -----------
        test_project: bool
            If true, the class creation will create a new project and
            store its GUID in the PROJECT_GUID_TEMPORARY variable
        """

        if test_project:
            self.PROJECT_GUID_TEMPORARY = (await self.http_client.post(
                "project",
                {
                    "name": "poc-train-simple-model",
                    "bucket_name": "ubiops-qognitive-default"
                }
            ))["guid"]
        self.project = await self._preload(
            "project",
            self.PROJECT_GUID_TEMPORARY
        )

    async def _preload(self, ep: str, guid: str) -> dict:
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
        return await self.http_client.get(f"{ep}/{guid}")

    async def _training_parameters(self, params: TrainingParameters) -> None:
        """
        Upload training parameters

        Parameters:
        -----------
        params: TrainingParameters
            Valid TypedDict of the training parameters
        """
        self.training_parameters = await self.http_client.post(
            "training_parameters",
            {
                "project_guid": self.project["guid"],
                "model": self.model.value,
                "parameters": {
                    "model": self.model.params
                } | params
            }
        )

    async def data(self, data: pd.DataFrame) -> AsyncQcogClient:
        """For a fresh "to train" model and properly initialized model
        upload a pandas DataFrame dataset.

        Parameters:
        -----------
        data: pd.DataFrame:
            the dataset as a DataFrame
        upload: bool:
            if true post the dataset

        Returns:
        --------
        AsyncQcogClient: itself
        """
        data_payload = Dataset(
            format="csv",
            source="client",
            data=encode_base64(data),
            project_guid=self.project["guid"],
        )
        valid_data: dict = {k: v for k, v in data_payload.items()}  # type cast
        self.dataset = await self.http_client.post("dataset", valid_data)
        return self

    async def preloaded_data(self, guid: str) -> AsyncQcogClient:
        """Retrieve a dataset that was previously uploaded from guid.

        Parameters:
        -----------
        guid: str:
            guid of a previously uploaded dataset

        Returns:
        --------
        AsyncQcogClient itself
        """
        self.dataset = await self._preload("dataset", guid)
        return self

    async def preloaded_training_parameters(
        self, guid: str,
        rebuild: bool = False
    ) -> AsyncQcogClient:
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
        AsyncQcogClient itself
        """
        self.training_parameters = await self._preload(
            "training_parameters",
            guid,
        )
        return self

    async def preloaded_model(self, guid: str) -> AsyncQcogClient:
        self.trained_model = await self._preload("model", guid)

        self.version = self.trained_model[
            "training_package_location"
        ].split("packages/")[-1].split("-")[1]

        await asyncio.gather(
            self._preload("project", self.trained_model["project_guid"]),
            self.preloaded_training_parameters(
                self.trained_model["training_parameters_guid"]
            )
        )

        model_params = {
            k.replace(
                "_kwargs", ""
            ): v for k, v in self.training_parameters[
                "parameters"
            ][
                "model"
            ].items()
            if k != "model"
        }

        self.model = MODEL_MAP[
            self.training_parameters["model"]
        ](**model_params)
        return self

    async def train(
        self,
        batch_size: int,
        num_passes: int,
        weight_optimization: NotRequiredWeightParams,
        get_states_extra: NotRequiredStateParams,
    ) -> AsyncQcogClient:
        """
        For a fresh "to train" model properly configured and initialized
        trigger a training request.

        Parameters:
        -----------
        params: TrainingParameters

        Returns:
        --------
        AsyncQcogClient: itself
        """

        params: TrainingParameters = TrainingParameters(
            batch_size=batch_size,
            num_passes=num_passes,
            weight_optimization_kwargs=weight_optimization,
            state_kwargs=get_states_extra,
        )

        await self._training_parameters(params)

        self.trained_model = await self.http_client.post(
            "model",
            {
                "training_parameters_guid": self.training_parameters["guid"],
                "dataset_guid": self.dataset["guid"],
                "project_guid": self.project["guid"],

                # TODO: we need to rewrite this to no expose internal details like that  # noqa: 503
                "training_package_location": f"s3://ubiops-qognitive-default/packages/qcog-{self.version}-cp310-cp310-linux_x86_64/training_package.zip"  # noqa: 503
            },
        )

        return self

    async def status(self) -> str:
        self.status_resp: dict = await self.http_client.get(
            f"model/{self.trained_model['guid']}"
        )

        self.last_status: str = self.status_resp["status"]
        return self.last_status

    async def wait_for_training(self, poll_time: int = 5) -> AsyncQcogClient:
        """
        Wait for training to complete.

        the function is blocking

        Parameters:
        -----------
        poll_time: int:
            status checks intervals in seconds

        Returns:
        --------
        AsyncQcogClient: itself

        """
        while (await self.status()) == "pending":
            if self.last_status == "completed":
                break
            await asyncio.sleep(poll_time)

        if self.last_status != "completed":
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
        pd.DataFrame: the predictions

        """
        self.inference_result: dict = await self.http_client.post(
            f"model/{self.trained_model['guid']}/inference",
            {
                "data": encode_base64(data),
                "parameters": parameters
            },
        )

        return base642dataframe(self.inference_result["response"]["data"])
