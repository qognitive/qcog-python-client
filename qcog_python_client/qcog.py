from __future__ import annotations

import pandas as pd

from .client import (
    RequestsClient,
    # decode_base64,  # for when inference returns dataframe
    base642dataframe,
    encode_base64,
)

from .model import (
    MODEL_MAP,
    Dataset,
    PauliModel,
    EnsembleModel,
    TrainProtocol,
    TrainingParameters,
    InferenceProtocol,
    InferenceParameters,
    Operator,
    NotRequiredWeightParams,
    NotRequiredStateParams,
)


def is_version_v1_gt_v2(v1: str, v2: str) -> bool:
    major1, minor1, fix1 = [int(w) for w in v1.split(".")]
    major2, minor2, fix2 = [int(w) for w in v2.split(".")]

    return major1 > major2 or minor1 > minor2 or fix1 > fix2


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
            raise ValueError(
                f"qcog version can't be older than {self.OLDEST_VERSION}"
            )
        if is_version_v1_gt_v2(version, self.NEWEST_VERSION):
            raise ValueError(
                f"qcog version can't be older than {self.NEWEST_VERSION}"
            )
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
                "training_package_location": f"s3://ubiops-qognitive-default/packages/qcog-{self.version}-cp310-cp310-linux_x86_64/training_package.zip"  # noqa: 503
            },
        )

        return self

    def status(self) -> str:
        resp: dict = self.http_client.get(
            f"model/{self.trained_model['guid']}"
        )

        status: str = resp["status"]
        return status

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
