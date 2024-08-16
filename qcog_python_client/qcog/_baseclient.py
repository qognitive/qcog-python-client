from __future__ import annotations

import asyncio
from typing import Any, TypeAlias

import aiohttp
import pandas as pd

from qcog_python_client.log import qcoglogger
from qcog_python_client.qcog._base64utils import base642dataframe
from qcog_python_client.qcog._data_uploader import encode_base64
from qcog_python_client.qcog._interfaces import ABCDataClient, ABCRequestClient
from qcog_python_client.qcog._jsonable_parameters import (
    jsonable_inference_parameters,
    jsonable_train_parameters,
)
from qcog_python_client.qcog._status import SUCCESS_STATUS, WAITING_STATUS
from qcog_python_client.qcog._version import DEFAULT_QCOG_VERSION, numeric_version
from qcog_python_client.qcog.pytorch.agent import PyTorchAgent
from qcog_python_client.schema import (
    InferenceParameters,
    TrainingParameters,
)
from qcog_python_client.schema.common import (
    Matrix,
    NotRequiredStateParams,
    NotRequiredWeightParams,
    PytorchTrainingParameters,
)
from qcog_python_client.schema.generated_schema.models import (
    AppSchemasDataPayloadDataPayloadResponse,
    AppSchemasParametersTrainingParametersPayloadResponse,
    AppSchemasPytorchModelPytorchModelPayloadResponse,
    AppSchemasTrainTrainedModelPayloadResponse,
    Model,
    ModelEnsembleParameters,
    ModelGeneralParameters,
    ModelPauliParameters,
    ModelPytorchParameters,
    TrainingStatus,
)
from tests.pytorch_model.model import train

Operator: TypeAlias = str | int
TrainingModel: TypeAlias = (
    ModelPauliParameters
    | ModelEnsembleParameters
    | ModelGeneralParameters
    | ModelPytorchParameters
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
        self._pytorch_model: dict | None = None

    @property
    def pytorch_model(self) -> dict:
        """Return the Pytorch model."""
        if self._pytorch_model is None:
            raise StateNotSetError(
                "No Pytorch model has been found associated with this request.",
                "You can use `pytorch` method to upload a new Pytorch model",
            )
        return self._pytorch_model

    @pytorch_model.setter
    def pytorch_model(self, value: dict) -> None:
        """Set and validate the Pytorch model."""
        # Pytorch model is the current model that has been set by the user
        # using the `pytorch` method. The database model is currently the
        # same as the `trained_model`, but it's not an actual trained model,
        # It's just a pointer to the uploaded model with some parameters,
        # and a specific dataset.
        self._pytorch_model = AppSchemasPytorchModelPytorchModelPayloadResponse.model_validate(  # noqa: E501
            value
        ).model_dump()

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
        print("-----> Validating : ", value)
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
        self._trained_model = AppSchemasTrainTrainedModelPayloadResponse.model_validate(
            value
        ).model_dump()

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
    # Model Parameters
    ############################
    def pauli(
        self,
        operators: list[Operator],
        qbits: int = 2,
        pauli_weight: int = 2,
        sigma_sq: dict[str, float] | None = None,
        sigma_sq_optimization: dict[str, float] | None = None,
        seed: int = 42,
        target_operator: list[Operator] | None = None,
    ) -> BaseQcogClient:
        """Select PauliModel for the training."""
        self._model = ModelPauliParameters(
            operators=[str(op) for op in operators],
            qbits=qbits,
            pauli_weight=pauli_weight,
            sigma_sq=sigma_sq or {},
            sigma_sq_optimization_kwargs=sigma_sq_optimization or {},
            seed=seed,
            target_operators=[str(op) for op in target_operator]
            if target_operator
            else [],
            model_name=Model.pauli.value,
        )
        return self

    def ensemble(
        self,
        operators: list[Operator],
        dim: int = 16,
        num_axes: int = 4,
        sigma_sq: dict[str, float] | None = None,
        sigma_sq_optimization: dict[str, float] | None = None,
        seed: int = 42,
        target_operator: list[Operator] | None = None,
    ) -> Any:
        """Select EnsembleModel for the training."""
        self._model = ModelEnsembleParameters(
            operators=[str(op) for op in operators],
            dim=dim,
            num_axes=num_axes,
            sigma_sq=sigma_sq or {},
            sigma_sq_optimization_kwargs=sigma_sq_optimization or {},
            seed=seed,
            target_operators=[str(op) for op in target_operator]
            if target_operator
            else [],
            model_name=Model.ensemble.value,
        )
        return self

    async def _pytorch(
        self,
        model_name: str,
        model_path: str,
    ) -> BaseQcogClient:
        """Select a Pythorch architecture defined by the user."""
        # Instantiate the Pytorch client.

        # Set the pytorch model parameters.
        # In this case there are no parameters
        # Because the parameters are defined
        # by the user in the model itself.
        self._model = ModelPytorchParameters(model_name=Model.pytorch.value)

        # Create a PyTorch agent with the http client functions
        agent = PyTorchAgent.create_agent()

        # Needed to upload the model and the parameters
        agent.register_tool("post_multipart", self._post_multipart)

        # Upload the model
        self.pytorch_model = await agent.upload_model(model_path, model_name)
        return self

    async def _data(self, data: pd.DataFrame) -> BaseQcogClient:
        """Upload Data."""
        # Delegating the upload function to the data client
        # So any change in the logic or service will not affect the client
        self.dataset = await self.data_client.upload_data(data)
        return self

    async def _preloaded_data(self, guid: str) -> BaseQcogClient:
        """Async method to retrieve a dataset that was previously uploaded from guid."""
        self.dataset = await self.http_client.get(f"dataset/{guid}")
        return self

    async def _preloaded_training_parameters(self, guid: str) -> BaseQcogClient:
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

    async def _preloaded_model(self, guid: str) -> BaseQcogClient:
        """Retrieve preexisting model payload."""
        self.trained_model = await self.http_client.get(f"model/{guid}")
        return self

    async def _train(
        self,
        batch_size: int,
        num_passes: int,
        weight_optimization: NotRequiredWeightParams,
        get_states_extra: NotRequiredStateParams,
    ) -> BaseQcogClient:
        """Start a training job."""
        params: TrainingParameters = TrainingParameters(
            batch_size=batch_size,
            num_passes=num_passes,
            weight_optimization_kwargs=weight_optimization,
            state_kwargs=get_states_extra,
        )

        await self._upload_training_parameters(params)

        accepted_response = await self.http_client.post(
            "model",
            {
                "training_parameters_guid": self.training_parameters["guid"],
                "dataset_guid": self.dataset["guid"],
                "qcog_version": self.version,
            },
        )

        self.trained_model = AppSchemasTrainTrainedModelPayloadResponse(
            guid=accepted_response["guid"],
            status=TrainingStatus.unknown,
            training_parameters_guid=accepted_response["training_parameters_guid"],
            loss=None,
            training_completion=0,
            current_batch_completion=0,
            qcog_version=accepted_response["qcog_version"],
            dataset_guid=accepted_response["dataset_guid"],
        ).model_dump()

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

        return base642dataframe(
            inference_result["response"]["data"],
        )

    async def _train_pytorch(
            self, training_parameters: PytorchTrainingParameters
        ) -> BaseQcogClient:
        agent = PyTorchAgent.create_agent()

        # Needed to upload the model and the parameters
        agent.register_tool("post_request", self.http_client.post)
        self.trained_model = await agent.train_model(
            self.pytorch_model["guid"],
            dataset_guid=self.dataset["guid"],
            training_parameters=training_parameters.model_dump(),
        )
        return self

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
        await self._load_trained_model()
        return {
            "guid": self.trained_model.get("guid"),
            "training_completion": self.trained_model.get("training_completion"),
            "current_batch_completion": self.trained_model.get(
                "current_batch_completion"
            ),
            "status": TrainingStatus(self.trained_model.get("status")),
        }

    async def _load_trained_model(self) -> None:
        """Load the status of the current trained model."""
        self.trained_model = await self.http_client.get(
            f"model/{self.trained_model['guid']}"
        )

    async def _status(self) -> TrainingStatus:
        """Check the status of the training job."""
        # Load last status
        await self._load_trained_model()

        self.last_status = TrainingStatus(self.trained_model["status"])
        training_completion = self.trained_model.get("training_completion", 0)
        current_batch_completion = self.trained_model.get("current_batch_completion", 0)

        logger.info(f"\nStatus: {self.last_status}")
        logger.info(f"Training completion: {training_completion} %")
        logger.info(f"Current batch completion: {current_batch_completion} %\n")

        self._loss = self.trained_model.get("loss", None)
        return self.last_status

    async def _get_loss(self) -> Matrix | None:
        """Return the loss matrix.

        Loss matrix is available only after training is completed.
        """
        # loss matrix is available only after training is completed
        if self._loss is None:
            await self._load_trained_model()
            self._loss = self.trained_model.get("loss", None)
        return self._loss

    async def _wait_for_training(self, poll_time: int = 60) -> None:
        """Wait for training to complete."""
        while await self._status() in WAITING_STATUS:
            if self.last_status in SUCCESS_STATUS:
                break
            await asyncio.sleep(poll_time)

        if self.last_status not in SUCCESS_STATUS:
            raise RuntimeError(f"Training failed. Status: {self.last_status}. ")

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

    async def _post_multipart(
        self,
        url: str,
        data: aiohttp.FormData,
    ) -> dict:
        # Add data headers to the form data
        return await self.http_client.post(
            url,
            data,
            content_type="data",
        )
