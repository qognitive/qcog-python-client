import asyncio
import json
from typing import Any, TypeAlias

import pandas as pd

from qcog_python_client.log import qcoglogger
from qcog_python_client.qcog._base64utils import base642dataframe
from qcog_python_client.qcog._data_uploader import DataClient, encode_base64
from qcog_python_client.qcog._interfaces import ABCDataClient, ABCRequestClient
from qcog_python_client.qcog._jsonable_parameters import (
    jsonable_inference_parameters,
    jsonable_train_parameters,
)
from qcog_python_client.qcog._status import SUCCESS_STATUS, WAITING_STATUS
from qcog_python_client.qcog._version import DEFAULT_QCOG_VERSION, numeric_version
from qcog_python_client.qcog.httpclient import RequestClient
from qcog_python_client.schema import (
    InferenceParameters,
    TrainingParameters,
)
from qcog_python_client.schema.common import (
    Matrix,
    NotRequiredStateParams,
    NotRequiredWeightParams,
)
from qcog_python_client.schema.generated_schema.models import (
    AcceptedResponse,
    AppSchemasDataPayloadDataPayloadResponse,
    AppSchemasParametersTrainingParametersPayloadResponse,
    Model,
    ModelEnsembleParameters,
    ModelGeneralParameters,
    ModelPauliParameters,
    TrainingStatus,
)

Operator: TypeAlias = str | int
TrainingModel: TypeAlias = (
    ModelPauliParameters | ModelEnsembleParameters | ModelGeneralParameters
)

logger = qcoglogger.getChild(__name__)


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


class BaseQcogClient:
    """Base Qcog Client."""

    def __init__(self) -> None:  # noqa: D107
        self.http_client: ABCRequestClient
        self.data_client: ABCDataClient
        self._version: str = DEFAULT_QCOG_VERSION
        self._model: TrainingModel | None = None
        self._project: dict | None = None
        self._dataset: dict | None = None
        self._training_parameters: dict | None = None
        self._trained_model: dict | None = None
        self._inference_result: dict | None = None
        self._loss: Matrix | None = None

    @property
    def model(self) -> TrainingModel:
        """Return the model."""
        if self._model is None:
            raise StateNotSetError(
                "model", "Please set the model first using `pauli` or `ensemble` method"
            )
        return self._model

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

    @dataset.setter
    def dataset(self, value: dict) -> None:
        """Set and validate the dataset."""
        self._dataset = AppSchemasDataPayloadDataPayloadResponse.model_validate(
            value
        ).model_dump()

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

    @training_parameters.setter
    def training_parameters(self, value: dict) -> None:
        """Set and validate the training parameters."""
        self._training_parameters = (
            AppSchemasParametersTrainingParametersPayloadResponse.model_validate(
                value
            ).model_dump()
        )

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

    @trained_model.setter
    def trained_model(self, value: dict) -> None:
        """Set and validate the trained model."""
        self._trained_model = AcceptedResponse.model_validate(value).model_dump()

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

    ############################
    # Init Method
    ############################
    @classmethod
    async def _create(
        cls,
        *,
        token: str | None = None,
        hostname: str = "dev.qognitive.io",
        port: int = 443,
        api_version: str = "v1",
        safe_mode: bool = False,
        version: str = DEFAULT_QCOG_VERSION,
        httpclient: ABCRequestClient | None = None,
        dataclient: ABCDataClient | None = None,
    ) -> "BaseQcogClient":
        """Create a new Qcog client.

        TODO: docstring
        """
        client = cls()
        client.version = version
        client.http_client = httpclient or RequestClient(
            token=token,
            hostname=hostname,
            port=port,
            api_version=api_version,
        )

        client.data_client = dataclient or DataClient(client.http_client)

        if safe_mode:
            await client.http_client.get("status")

        return client

    ############################
    # Model Parameters
    ############################
    def pauli(
        self,
        operators: list[Operator],
        qbits: int = 2,
        pauli_weight: int = 2,
        sigma_sq: dict[str, float] = {},
        sigma_sq_optimization: dict[str, float] = {},
        seed: int = 42,
        target_operator: list[Operator] = [],
    ) -> "BaseQcogClient":
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

    async def _data(self, data: pd.DataFrame) -> "BaseQcogClient":
        """Upload Data."""
        # Delegating the upload function to the data client
        # So any change in the logic or service will not affect the client
        self.dataset = await self.data_client.upload_data(data)
        return self

    async def _preloaded_data(self, guid: str) -> "BaseQcogClient":
        """Async method to retrieve a dataset that was previously uploaded from guid."""
        self.dataset = await self.http_client.get(f"dataset/{guid}")
        return self

    async def _preloaded_training_parameters(self, guid: str) -> "BaseQcogClient":
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
        self.training_parameters = await self.http_client.get(
            f"training_parameters/{guid}"
        )
        return self

    async def _preloaded_model(self, guid: str) -> "BaseQcogClient":
        """Retrieve preexisting model payload."""
        self.trained_model = await self.http_client.get(f"model/{guid}")
        return self

    async def _train(
        self,
        batch_size: int,
        num_passes: int,
        weight_optimization: NotRequiredWeightParams,
        get_states_extra: NotRequiredStateParams,
    ) -> "BaseQcogClient":
        """Start a training job."""
        params: TrainingParameters = TrainingParameters(
            batch_size=batch_size,
            num_passes=num_passes,
            weight_optimization_kwargs=weight_optimization,
            state_kwargs=get_states_extra,
        )

        await self._upload_training_parameters(params)

        self.trained_model = await self.http_client.post(
            "model",
            {
                "training_parameters_guid": self.training_parameters["guid"],
                "dataset_guid": self.dataset["guid"],
                "qcog_version": self.version,
            },
        )

        return self

    async def _inference(
        self,
        data: pd.DataFrame,
        parameters: InferenceParameters,
    ) -> pd.DataFrame:
        """From a trained model query an inference."""
        inference_result = await self.http_client.post(
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

    ############################
    # Public Utilities Methods #
    ############################
    async def _progress(self) -> dict:
        """Return the current status of the training.

        Returns
        -------
        dict
            the current status of the training
            `guid` : str
            `training_completion` : int
            `current_batch_completion` : int
            `status` : TrainingStatus

        """
        self._trained_model = await self.http_client.get(
            f"model/{self.trained_model['guid']}"
        )
        return {
            "guid": self.trained_model.get("guid"),
            "training_completion": self.status_resp.get("training_completion"),
            "current_batch_completion": self.trained_model.get(
                "current_batch_completion"
            ),
            "status": TrainingStatus(self.trained_model.get("status")),
        }

    async def _status(self) -> TrainingStatus:
        """Check the status of the training job."""
        self.status_resp: dict = await self.http_client.get(
            f"model/{self.trained_model['guid']}"
        )
        status = TrainingStatus(self.status_resp["status"])
        training_completion = self.status_resp.get("training_completion", 0)
        current_batch_completion = self.status_resp.get("current_batch_completion", 0)

        logger.info(f"\nStatus: {status}")
        logger.info(f"Training completion: {training_completion} %")
        logger.info(f"Current batch completion: {current_batch_completion} %\n")

        self.last_status = TrainingStatus(self.status_resp["status"])
        self._loss = self.status_resp.get("loss", None)

        return self.last_status

    async def _get_loss(self) -> Matrix | None:
        """Return the loss matrix.

        Loss matrix is available only after training is completed.
        """
        # loss matrix is available only after training is completed
        if self._loss is None:
            # TODO: Validate response
            self.status_resp = await self.http_client.get(
                f"model/{self.trained_model['guid']}"
            )
            self.last_status = TrainingStatus(self.status_resp["status"])
            self._loss = self.status_resp.get("loss", None)
        return self._loss

    async def _wait_for_training(self, poll_time: int = 60) -> None:
        """Wait for training to complete."""
        while await self._status() in WAITING_STATUS:
            if self.last_status in SUCCESS_STATUS:
                break
            await asyncio.sleep(poll_time)

        if self.last_status not in SUCCESS_STATUS:
            raise RuntimeError(
                f"Training failed: {json.dumps(self.status_resp, indent=4)}"
            )

    ############################
    # Private Utility Methods #
    ############################

    async def _upload_training_parameters(self, params: TrainingParameters) -> None:
        """Upload Training Parameters."""
        self.training_parameters = await self.http_client.post(
            "training_parameters",
            {
                "model": self.model.model_name,
                "parameters": {"model": self.model.model_dump()}
                | jsonable_train_parameters(params),
            },
        )
