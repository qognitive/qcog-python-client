import os
import requests


class QcogClient:

    TOKEN: str = os.environ.get("QCOG_API_TOKEN", "N/A")
    HOSTNAME: str = os.environ.get("QCOG_HOSTNAME", "0.0.0.0")
    PORT: str = os.environ.get("QCOG_PORT", "443")
    VERSION: str = "v1"
    PROJECT_GUID_TEMPORARY: str = "45ec9045-3d50-46fb-a82c-4aa0502801e9"

    def __init__(
        self,
        token: str | None = None,
        hostname: str | None = None,
        port: str | int | None = None,
        secure: bool = True,
        safe_mode: bool = True,  # NOTE will make False default later
        test_project: bool = False,
        verify: bool = True,  # for debugging until ssl is fixed
    ):
        self.token: str = token if isinstance(token, str) else self.TOKEN
        if self.token == "N/A":
            raise RuntimeError("missing token")

        self.hostname = hostname if isinstance(
            hostname, str
        ) else self.HOSTNAME
        self.port = str(port) if isinstance(
            port, str | int
        ) else self.PORT

        self.headers = {
            "Authorization": f"Bearer {self.token}"
        }
        prefix: str = "https://" if secure else "http://"
        base_url: str = f"{prefix}{self.hostname}:{self.port}"
        self.url: str = f"{base_url}/api/v1"
        self.checks: list[str] = [
            f"{base_url}/status/",
            f"{base_url}/health/db/",
            f"{base_url}/health/s3/",
        ]
        self.safe_mode: bool = safe_mode
        self.verify: bool = verify

        self._test_connection()
        self._resolve_project(test_project)

    def _get(self, uri: str) -> requests.Response:
        resp = requests.get(uri, headers=self.headers, verify=self.verify)

        try:
            resp.raise_for_status()
        except Exception as e:
            print(resp.status_code)
            print(resp.text)
            raise e

        return resp

    def _post(self, uri: str, data: dict) -> requests.Response:
        resp = requests.post(uri, headers=self.headers, json=data, verify=self.verify)

        try:
            resp.raise_for_status()
        except Exception as e:
            print(resp.status_code)
            print(resp.text)
            raise e

        return resp

    def _test_connection(self) -> None:
        if self.safe_mode:
            for uri in self.checks:
                self._get(uri)

    def _resolve_project(self, test_project: bool) -> None:

        if test_project:
            self.PROJECT_GUID_TEMPORARY = self.post(
                "project",
                {
                    "name": "poc-train-simple-model",
                    "bucket_name": "ubiops-qognitive-default"
                }
            )["guid"]

        resp = self._get(f"{self.url}/project/{self.PROJECT_GUID_TEMPORARY}/")
        self.project: dict[str, str] = resp.json()

    def get(self, endpoint: str) -> dict:
        retval: dict = self._get(f"{self.url}/{endpoint}/").json()
        return retval

    def post(self, endpoint: str, data: dict) -> dict:
        retval: dict = self._post(f"{self.url}/{endpoint}/", data=data).json()
        return retval
