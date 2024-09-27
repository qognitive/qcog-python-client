"""Async wrapper."""

from __future__ import annotations

from typing import Any, overload

import pandas as pd

from qcog_python_client.qcog._baseclient import (
    BaseQcogClient,
)
from qcog_python_client.qcog._data_uploader import DataClient
from qcog_python_client.qcog._httpclient import RequestClient
from qcog_python_client.qcog._interfaces import (
    IDataClient,
    IRequestClient,
)
from qcog_python_client.qcog._version import DEFAULT_QCOG_VERSION
from qcog_python_client.schema.common import (
    InferenceParameters,
    Matrix,
    NotRequiredStateParams,
    NotRequiredWeightParams,
    PytorchTrainingParameters,
)
from qcog_python_client.schema.generated_schema.models import TrainingStatus


class AsyncQcogClient(BaseQcogClient):
    """Async Qcog Client."""

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
        httpclient: IRequestClient | None = None,
        dataclient: IDataClient | None = None,
    ) -> AsyncQcogClient:
        """Instantiate a new Qcog client.

        This client is meant to work in an async context.

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
        httpclient : ABCRequestClient | None
            an optional http client to use instead of the default
        dataclient : ABCDataClient | None
            an optional data client to use instead of the default

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

    async def data(self, data: pd.DataFrame) -> AsyncQcogClient:
        """Upload a dataset.

        Parameters
        ----------
        data : pd.DataFrame
            the dataset as a DataFrame

        Returns
        -------
        QcogClient

        """
        await self._data(data)
        return self

    async def upload_data(self, data: pd.DataFrame, dataset_id: str) -> AsyncQcogClient:
        """Upload data as a stream."""
        await self._upload_data(data, dataset_id)
        return self

    async def preloaded_data(self, guid: str) -> AsyncQcogClient:
        """Retrieve a dataset that was previously uploaded from guid.

        Parameters
        ----------
        guid : str
            guid of a previously uploaded dataset

        Returns
        -------
        QcogClient

        """
        await self._preloaded_data(guid)
        return self

    async def preloaded_training_parameters(self, guid: str) -> AsyncQcogClient:
        """Retrieve preexisting training parameters payload.

        Parameters
        ----------
        guid : str
            model guid

        Returns
        -------
        QcogClient
            itself

        """
        await self._preloaded_training_parameters(guid)
        return self

    @overload
    async def preloaded_model(self, guid: str) -> AsyncQcogClient: ...

    @overload
    async def preloaded_model(
        self,
        guid: str,
        *,
        pytorch_model_name: str | None = None,
    ) -> AsyncQcogClient: ...

    @overload
    async def preloaded_model(
        self, *, pytorch_model_name: str | None = None, force_reload: bool = False
    ) -> AsyncQcogClient: ...

    async def preloaded_model(
        self,
        guid: str | None = None,
        *,
        pytorch_model_name: str | None = None,
        force_reload: bool = False,
    ) -> AsyncQcogClient:
        """Retrieve a preexisting trained model.

        Parameters
        ----------
        guid : str | None
            trained model identifier. If you are working on a pytorch model,
            you also need to run `preload_pt_model` to load the model architecture,
            or you can provide `pytorch_model_name` parameter with the name
            of the model. If no `guid` is provided and you are working with
            a pytorch model, the client will try to load the latest trained model.
        pytorch_model_name : str | None
            the name of the PyTorch model. This is the identifier that you
            used when you uploaded the model using `pytorch` method.
            It should be provided if no model architecture is loaded.
        force_reload : bool | None
            If true will fetch the latest models at every request.

        Returns
        -------
        QcogClient

        """
        # if a guid is provided, just fetch the model
        await self._preloaded_model(
            guid,
            pytorch_model_name=pytorch_model_name,
            force_reload=force_reload,
        )
        return self

    async def preloaded_pt_model(self, model_name: str) -> AsyncQcogClient:
        """Retrieve a preexisting PyTorch model.

        Parameters
        ----------
        model_name : str
            model name

        Returns
        -------
        QcogClient

        """
        await self._preloaded_pt_model(model_name)
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
        await self._train(batch_size, num_passes, weight_optimization, get_states_extra)
        return self

    async def inference(
        self, data: pd.DataFrame, parameters: InferenceParameters | None
    ) -> pd.DataFrame | Any:
        """From a trained model query an inference.

        Parameters
        ----------
        data : pd.DataFrame
            the dataset as a DataFrame
        parameters : dict
            inference parameters

        Returns
        -------
        pd.DataFrame | Any
            the inference result

        """
        return await self._inference(data, parameters)

    async def train_pytorch(self, training_parameters: dict) -> AsyncQcogClient:
        """Train PyTorch model.

        Run a training session for a PyTorch model.
        The training session should be run against a valid Pytorch model and dataset
        previously selected.

        Use `.data(...)` and `.pytorch(...)` to set the model and dataset.

        Parameters
        ----------
        training_parameters : dict
            the training parameters as specified in the `train`
            function of the provided model

        Returns
        -------
        AsyncQcogClient

        """
        await super()._train_pytorch(
            PytorchTrainingParameters.model_validate(training_parameters),
        )
        return self

    # ###########################
    # Public Models
    # ###########################
    def pauli(self, *args: Any, **kwargs: Any) -> AsyncQcogClient:  # noqa: D417
        """Select Pauli model.

        Parameters
        ----------
        operators: list[str | int]
            List of operators

        qbits: int, default=2
            Number of qbits

        pauli_weight: int, default=2
            Pauli weight

        sigma_sq: dict, default=None
            Sigma squared

        sigma_sq_optimization: dict, default=None
            Sigma squared optimization

        seed: int, default=42
            Seed

        target_operator: list[str | int], default=None
            Target operator


        Returns
        -------
        AsyncQcogClient

        """
        super().pauli(*args, **kwargs)
        return self

    def ensemble(self, *args: Any, **kwargs: Any) -> AsyncQcogClient:  # noqa: D417
        """Select Ensemble model.

        Parameters
        ----------
        operators: list[str | int]
            List of operators

        dim: int, default=16
            Dimension

        num_axes: int, default=4
            Number of axes

        sigma_sq: dict, default=None
            Sigma squared

        sigma_sq_optimization: dict, default=None
            Sigma squared optimization

        seed: int, default=42
            Seed

        target_operator: list[str | int], default=None
            Target operator

        Returns
        -------
        AsyncQcogClient

        """
        super().ensemble(*args, **kwargs)
        return self

    async def pytorch(
        self,
        model_name: str,
        model_path: str,
    ) -> AsyncQcogClient:
        """Select PyTorch model.

        Parameters
        ----------
        model_name : str
            the name of the model
        model_path : str
            the path to the model

        Returns
        -------
        AsyncQcogClient

        """
        await super()._pytorch(
            model_name,
            model_path,
        )
        return self

    # ###########################
    # Public Utilities
    # ###########################
    async def progress(self) -> dict:
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
        return await self._progress()

    async def status(self) -> TrainingStatus:
        """Return the current status of the training.

        Returns
        -------
        TrainingStatus
            the current status of the training

        """
        return await self._status()

    async def get_loss(self) -> Matrix | None:
        """Return the loss matrix.

        Returns
        -------
        Matrix
            the loss matrix

        """
        return await self._get_loss()

    async def wait_for_training(self, poll_time: int = 60) -> None:
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
        await self._wait_for_training(poll_time)
