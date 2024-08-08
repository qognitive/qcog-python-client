"""Sync wrapper for the QcogClient."""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

import pandas as pd

from qcog_python_client.qcog._baseclient import BaseQcogClient
from qcog_python_client.qcog._data_uploader import DataClient
from qcog_python_client.qcog._interfaces import (
    ABCDataClient,
    ABCRequestClient,
)
from qcog_python_client.qcog._version import DEFAULT_QCOG_VERSION
from qcog_python_client.qcog.httpclient import RequestClient
from qcog_python_client.schema.common import (
    InferenceParameters,
    Matrix,
    NotRequiredStateParams,
    NotRequiredWeightParams,
)
from qcog_python_client.schema.generated_schema.models import TrainingStatus

CallableReturnType = TypeVar("CallableReturnType")


class QcogClient(BaseQcogClient):
    """Qcog Client."""

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
        httpclient: ABCRequestClient | None = None,
        dataclient: ABCDataClient | None = None,
    ) -> "BaseQcogClient":
        """Create a new Qcog client.

        TODO: docstring
        """
        client = cls()
        client.version = version
        client._http_client = httpclient or RequestClient(
            token=token,
            hostname=hostname,
            port=port,
            api_version=api_version,
        )

        client._data_client = dataclient or DataClient()

        if safe_mode:
            cls.await_async(client.http_client.get("status"))

        return client

    def data(self, data: pd.DataFrame) -> QcogClient:
        """Create a new instance of the client."""
        self.await_async(self._data(data))
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

        """
        self.await_async(self._preloaded_data(guid))
        return self

    def preloaded_training_parameters(self, guid: str) -> QcogClient:
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
        self.await_async(self._preloaded_training_parameters(guid))
        return self

    def preload_model(self, guid: str) -> QcogClient:
        """Retrieve a preexisting model.

        Parameters
        ----------
        guid : str
            model guid

        Returns
        -------
        QcogClient

        """
        self.await_async(self._preloaded_model(guid))
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
        self.await_async(
            self._train(batch_size, num_passes, weight_optimization, get_states_extra)
        )
        return self

    def inference(
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
        return self.await_async(self._inference(data, parameters))

    # ###########################
    # Public Utilities
    # ###########################
    def progress(self) -> dict:
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
        return self.await_async(self._progress())

    def status(self) -> TrainingStatus:
        """Return the current status of the training.

        Returns
        -------
        TrainingStatus
            the current status of the training

        """
        return self.await_async(self._status())

    def get_loss(self) -> Matrix | None:
        """Return the loss matrix.

        Returns
        -------
        Matrix
            the loss matrix

        """
        return self.await_async(self._get_loss())

    def wait_for_training(self) -> None:
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
        self.await_async(self._wait_for_training())

    def await_async(
        self, async_callable: Coroutine[Any, Any, CallableReturnType]
    ) -> CallableReturnType:
        """Await an async callable."""
        # If the function is running inside an event loop, get the event loop
        # and run the coroutine.
        loop = asyncio.get_event_loop()

        if loop.is_running():
            return loop.run_until_complete(async_callable)

        # If the function is not running inside an event loop, create a new one
        # in its own thread and run the coroutine.

        # Create a new event loop
        try:
            new_loop = asyncio.new_event_loop()
            # Create the executor
            # executor = futures.ThreadPoolExecutor(max_workers=10)
            # executor.submit(new_loop.run_forever)
            asyncio.set_event_loop(new_loop)
            future = asyncio.run_coroutine_threadsafe(async_callable, new_loop)
            return future.result()
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)
