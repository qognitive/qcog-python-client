from __future__ import annotations

import base64
import io
import os
import time
from enum import Enum

import aiohttp
import requests
import random

import pandas as pd


class HttpMethod(Enum):
    get = "get"
    post = "post"


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


def base642dataframe(encoded_string: str, indexing: list[int]) -> pd.DataFrame:
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
    return pd.read_csv(s, index_col=indexing)


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
    raw_string: str = data.to_csv()
    raw_bytes: bytes = raw_string.encode("ascii")
    base64_bytes = base64.b64encode(raw_bytes)
    base64_string = base64_bytes.decode("ascii")
    return base64_string


class _HTTPClient:
    """
    This class is the https API client
    """

    TOKEN: str = os.environ.get("QCOG_API_TOKEN", "")
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
        verify: bool = True,  # for debugging until ssl is fixed
        retries: int = 3
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
        verify: bool
            ignore ssl provenance for testing purposes
        retries: int
            number of attempts in cases of bad gateway
        """

        self.token: str = token if isinstance(token, str) else self.TOKEN
        if not self.token:
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
        self.verify: bool = verify
        self.retries: int = retries


class RequestsClient(_HTTPClient):
    """This class is the synchronous implementation of the API client."""

    def _request_retry(
        self,
        uri: str,
        method: HttpMethod,
        data: dict | None = None,
    ) -> requests.Response:
        """Execute the get "requests" by adding class-level settings

        Parameters:
        -----------
        uri: str
            Full http url
        data: dict
            in case of post data to post otherwise empty dict
        method: HttpMethod
            method enum

        Returns:
        --------
        requests.Response object
            will raise_for_status so caller
            may use .json()
        """
        random.seed()
        sleep_for: int = random.randrange(1, 5)
        exception: Exception

        for retry in range(self.retries):
            try:

                resp = requests.request(
                    method.value,
                    uri,
                    json=data,
                    headers=self.headers,
                    verify=self.verify
                )
                resp.raise_for_status()

                return resp

            except Exception as e:

                time.sleep(sleep_for)
                sleep_for = random.randrange(1, 2 * sleep_for)
                exception = e

        print(resp.status_code)
        print(resp.text)

        raise exception

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
        retval: dict = self._request_retry(
            f"{self.url}/{endpoint}/",
            HttpMethod.get,
        ).json()

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
        retval: dict = self._request_retry(
            f"{self.url}/{endpoint}/",
            HttpMethod.post,
            data,
        ).json()

        return retval


class AIOHTTPClient(_HTTPClient):
    """This class is the async implementation of the API client"""

    async def _request_retry(
        self,
        uri: str,
        method: HttpMethod,
        data: dict | None = None,
    ) -> dict:
        """
        Execute the async get "aiohttp" by adding class-level settings

        Parameters:
        -----------
        uri: str
            Full http url
        data: dict
            in case of post, the posted data, empty dict otherwise
        method: HttpMethod
            request type enum

        Returns:
        --------
        FIXME.Response object
            will raise_for_status so caller
            may use .json()
        """
        random.seed()
        sleep_for: int = random.randrange(1, 5)
        exception: aiohttp.client_exceptions.ClientResponseError

        for retry in range(self.retries):
            try:
                async with aiohttp.ClientSession(
                    headers=self.headers,
                    raise_for_status=True
                ) as session:

                    resp = await session.request(
                        method.value,
                        uri,
                        json=data,
                        ssl=self.verify,
                    )
                    retval: dict = await resp.json()

                    return retval

            except aiohttp.client_exceptions.ClientResponseError as e:

                time.sleep(sleep_for)
                sleep_for = random.randrange(1, 2 * sleep_for)
                exception = e

        raise exception

    async def get(self, endpoint: str) -> dict:
        """Convenience wrapper around aiohttp.get (called via _get method)

        Parameters:
        -----------
        endpoint: str
            a valid prefix to the orchestration API (including guid
            if applicable) and will add to the dns prefix

        Returns:
        --------
            dict: unpacked json dict
        """
        return await self._request_retry(
            f"{self.url}/{endpoint}/",
            HttpMethod.get,
        )

    async def post(self, endpoint: str, data: dict) -> dict:
        """Convenience wrapper around requests.post (called via _post method)

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
        return await self._request_retry(
            f"{self.url}/{endpoint}/",
            HttpMethod.post,
            data,
        )
