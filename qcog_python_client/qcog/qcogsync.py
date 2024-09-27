"""Sync wrapper for the QcogClient."""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

import pandas as pd

from qcog_python_client.qcog._baseclient import BaseQcogClient
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
        httpclient: IRequestClient | None = None,
        dataclient: IDataClient | None = None,
    ) -> QcogClient:
        """Instantiate a new Qcog client.

        This client is meant to work in a synchronous context.
        It will raise an error if used in an async context.

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

        client.data_client = dataclient or DataClient(http_client=client.http_client)

        if safe_mode:
            cls.await_async(client, client.http_client.get("status"))

        return client

    def data(self, data: pd.DataFrame) -> QcogClient:
        """Upload a dataset.

        Parameters
        ----------
        data : pd.DataFrame
            the dataset as a DataFrame

        Returns
        -------
        QcogClient

        """
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

    def preloaded_model(self, guid: str) -> QcogClient:
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
        pd.DataFrame
            the predictions

        """
        return self.await_async(self._inference(data, parameters))

    # ###########################
    # Public Models
    # ###########################
    def pauli(self, *args: Any, **kwargs: Any) -> QcogClient:  # noqa: D417
        """Pauli model.

        Select Pauli model.

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
        QcogClient

        """
        super().pauli(*args, **kwargs)
        return self

    def ensemble(self, *args: Any, **kwargs: Any) -> QcogClient:  # noqa: D417
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

    def wait_for_training(self, poll_time: int = 60) -> None:
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
        self.await_async(self._wait_for_training(poll_time))

    def await_async(
        self, async_callable: Coroutine[Any, Any, CallableReturnType]
    ) -> CallableReturnType:
        """Await an async callable."""
        # Make sure the function is not running in an event loop
        try:
            asyncio.get_running_loop()
            raise SystemError(
                "Cannot run async code in an event loop. "
                "Please run this code in a sync context."
            )
        except RuntimeError:
            # No event loop, run the async code
            return asyncio.run(async_callable)
