from __future__ import annotations

import base64
import io
import os
import requests

import pandas as pd


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


class RequestsClient:
    """
    This class is the https API client
    """

    TOKEN: str = os.environ.get("QCOG_API_TOKEN", "N/A")
    HOSTNAME: str = os.environ.get("QCOG_HOSTNAME", "api.qognitive.io")
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
        """
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
        """

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
