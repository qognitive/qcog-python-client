from __future__ import annotations

import base64
import io
import os
import requests

import pandas as pd

from .model import (
    MODEL_MAP,
    Dataset,
    PauliModel,
    EnsembleModel,
    TrainProtocol,
    TrainingParameters,
    InferenceProtocol,
    Operator,
    NotRequiredWeightParams,
    NotRequiredStateParams,
)


def decode_base64(encoded_string: str) -> str:
    """
    From a base64 encoded str type, decode into original
    string str type

    Parameters:
    -----------
    encoded_string: str
        encoded base64 string

    Returns:
    --------
    str: decoded string
    """
    base64_bytes: bytes = encoded_string.encode("ascii")
    decoded_bytes: bytes = base64.b64decode(base64_bytes)
    return decoded_bytes.decode("ascii")


def base642dataframe(encoded_string: str) -> pd.DataFrame:
    """
    From a base64 encoded str type, decode into original
    string str type and parse as csv dataframe using io

    Parameters:
    -----------
    encoded_string: str
        encoded base64 string

    Returns:
    --------
    pd.DataFrame: parsed csv dataframe
    """
    decoded_string: str = decode_base64(encoded_string)
    s = io.StringIO(decoded_string)
    return pd.read_csv(s)


def encode_base64(data: pd.DataFrame) -> str:
    """
    take a normal pandas dataframe and encode as
    base64 "string" of csv export

    Parameters:
    -----------
    data: pd.DataFrame

    Returns:
    str: encoded base64 string
    """
    raw_string: str = data.to_csv(index=False)
    raw_bytes: bytes = raw_string.encode("ascii")
    base64_bytes = base64.b64encode(raw_bytes)
    base64_string = base64_bytes.decode("ascii")
    return base64_string


def is_version_v1_gt_v2(v1, v2):
    major1, minor1, fix1 = [int(w) for w in v1.split(".")]
    major2, minor2, fix2 = [int(w) for w in v2.split(".")]

    return major1 > major2 or minor1 > minor2 or fix1 > fix2


class RequestsClient:
    """
    This class is the https API client
    """

    TOKEN: str = os.environ.get("QCOG_API_TOKEN", "N/A")
    HOSTNAME: str = os.environ.get("QCOG_HOSTNAME", "0.0.0.0")
    PORT: str = os.environ.get("QCOG_PORT", "443")

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
    ):

        self.token: str = token if isinstance(token, str) else self.TOKEN
        if self.token == "N/A":
            raise RuntimeError("missing token")

        self.hostname: str = hostname if isinstance(
            hostname, str
        ) else self.HOSTNAME
        self.port: str | int = str(port) if isinstance(
            port, str | int
        ) else self.PORT
        self.api_version: str = api_version

        self.headers = {
            "Authorization": f"Bearer {self.token}"
        }
        prefix: str = "https://" if secure else "http://"
        base_url: str = f"{prefix}{self.hostname}:{self.port}"
        self.url: str = f"{base_url}/api/{self.api_version}"
        self.checks: list[str] = [
            f"{base_url}/status/",
            f"{base_url}/health/db/",
            f"{base_url}/health/s3/",
        ]
        self.safe_mode: bool = safe_mode
        self.verify: bool = verify

        self._test_connection()

    def _get(self, uri: str) -> requests.Response:
        """
        Execute the get "requests" by adding class-level settings

        Parameters:
        -----------
        uri: str
            Full http url

        Returns:
        --------
        requests.Response object
            will raise_for_status so caller
            may use .json()
        """
        resp = requests.get(uri, headers=self.headers, verify=self.verify)

        try:
            resp.raise_for_status()
        except Exception as e:
            print(resp.status_code)
            print(resp.text)
            raise e

        return resp

    def _post(self, uri: str, data: dict) -> requests.Response:
        """
        Execute the posts "requests" by adding class-level settings

        Parameters:
        -----------
        uri: str
            Full http url
        data: dict
            json-able data payload

        Returns:
        --------
        requests.Response object
            will raise_for_status so caller
            may use .json()
        """
        resp = requests.post(
            uri,
            headers=self.headers,
            json=data,
            verify=self.verify,
        )

        try:
            resp.raise_for_status()
        except Exception as e:
            print(resp.status_code)
            print(resp.text)
            raise e

        return resp

    def _test_connection(self) -> None:
        """
        Run health checks at class creation
        """
        if self.safe_mode:
            for uri in self.checks:
                self._get(uri)

    def get(self, endpoint: str) -> dict:
        """
        Convenience wrapper around requests.get (called via _get method)

        Parameters:
        -----------
        endpoint: str
            a valid prefix to the orchestration API (including guid
            if applicable) and will add to the dns prefix

        Returns:
        --------
            dict: unpacked json dict
        """
        retval: dict = self._get(f"{self.url}/{endpoint}/").json()
        return retval

    def post(self, endpoint: str, data: dict) -> dict:
        """
        Convenience wrapper around requests.post (called via _post method)

        Parameters:
        -----------
        endpoint: str
            a valid prefix to the orchestration API (including guid
            if applicable) and will add to the dns prefix
        data: dict
            json-able data payload

        Returns:
        --------
            dict: unpacked json dict
        """
        retval: dict = self._post(
            f"{self.url}/{endpoint}/",
            data=data
        ).json()
        return retval


class QcogClient(TrainProtocol, InferenceProtocol):

    OLDEST_VERSION = "0.0.43"
    NEWEST_VERSION = "0.0.44"
    PROJECT_GUID_TEMPORARY: str = "45ec9045-3d50-46fb-a82c-4aa0502801e9"

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
        version: str = NEWEST_VERSION,
    ):

        self.http_client = RequestsClient(
            token=token,
            hostname=hostname,
            port=port,
            api_version=api_version,
            secure=secure,
            safe_mode=safe_mode,
            verify=verify,
        )
        self.model: PauliModel | EnsembleModel
        if is_version_v1_gt_v2(self.OLDEST_VERSION, version):
             raise ValueError(f"qcog version can't be older than {self.OLDEST_VERSION}")
        if is_version_v1_gt_v2(version, self.NEWEST_VERSION):
             raise ValueError(f"qcog version can't be older than {self.NEWEST_VERSION}")
        self.version: str = version
        self.project: dict[str, str]
        self.dataset: dict = {}
        self.training_parameters: dict = {}
        self.trained_model: dict = {}
        self.inference_result: dict = {}
        self._resolve_project(test_project)

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

    def pauli(
        self,
        operators: list[Operator],
        qbits: int = 2,
        pauli_weight: int = 2,
        sigma_sq: dict[str, float] = {},
        sigma_sq_optimization: dict[str, float] = {},
        seed: int = 42,
        target_operator: list[Operator] = [],
    ) -> QcogClient:
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
    ) -> QcogClient:
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

        self.preloaded_training_parameters(self.trained_model["training_parameters_guid"])

        model_params = {
            k.replace("_kwargs", ""): v for k, v in self.training_parameters["parameters"]["model"].items()
            if k != "model"
        }

        self.model = MODEL_MAP[self.training_parameters["model"]](**model_params)
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
                "training_package_location": f"s3://ubiops-qognitive-default/packages/qcog-{self.version}-cp310-cp310-linux_x86_64/training_package.zip"  # noqa: 503
            },
        )

        return self

    def status(self) -> dict:  # TODO extract the string
        return self.http_client.get(f"model/{self.trained_model['guid']}")

    def inference(
        self,
        data: pd.DataFrame,
        operators_to_forcast: list[Operator],
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
                "operators_to_forecast": operators_to_forcast
            },
        )

        return base642dataframe(self.inference_result["response"]["data"])
