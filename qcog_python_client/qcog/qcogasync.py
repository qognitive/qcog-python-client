"""Async wrapper."""

from __future__ import annotations

from typing import Any

import pandas as pd

from qcog_python_client.qcog._baseclient import BaseQcogClient
from qcog_python_client.qcog._data_uploader import DataClient
from qcog_python_client.qcog._httpclient import RequestClient
from qcog_python_client.qcog._interfaces import (
    ABCDataClient,
    ABCRequestClient,
)
from qcog_python_client.qcog._version import DEFAULT_QCOG_VERSION
from qcog_python_client.schema.common import (
    InferenceParameters,
    Matrix,
    NotRequiredStateParams,
    NotRequiredWeightParams,
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
        httpclient: ABCRequestClient | None = None,
        dataclient: ABCDataClient | None = None,
    ) -> AsyncQcogClient:
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

    async def preloaded_model(self, guid: str) -> AsyncQcogClient:
        """Retrieve a preexisting model.

        Parameters
        ----------
        guid : str
            model guid

        Returns
        -------
        QcogClient

        """
        await self._preloaded_model(guid)
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
        self, data: pd.DataFrame, parameters: InferenceParameters
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
        return await self._inference(data, parameters)

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
