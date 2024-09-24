"""HTTP API client for Qognitive."""

from __future__ import annotations

import asyncio
import os
import random
from enum import Enum

import aiohttp
import urllib3
import urllib3.util

from qcog_python_client.log import qcoglogger as logger
from qcog_python_client.qcog._interfaces import ABCRequestClient


class HttpMethod(Enum):
    """HTTP method enum."""

    get = "get"
    post = "post"


class _HTTPClient:
    """HTTPS API client."""

    TOKEN: str = os.environ.get("QCOG_API_TOKEN", "")

    def __init__(
        self,
        *,
        token: str | None = None,
        hostname: str = "dev.qognitive.io",
        port: int = 443,
        api_version: str = "v1",
        retries: int = 3,
    ):
        """HTTP client constructor.

        Parameters
        ----------
        token : str | None
            A valid API token granting access optional
            when unset (or None) expects to find the proper
            value as QCOG_API_TOKEN environment veriable
        hostname : str | None
            optional string of the hostname. Currently default
            to a standard api endpoint
        port : str | int | None
            port value default to https 443
        api_version : str
            the "vX" part of the url for the api version
        retries: int
            number of attempts in cases of bad gateway

        """
        self.token: str = token if isinstance(token, str) else self.TOKEN
        if not self.token:
            raise RuntimeError("missing token")

        self.hostname: str = hostname
        self.port: int = port
        self.api_version: str = api_version

        self.headers = {"Authorization": f"Bearer {self.token}"}
        protocol = "http" if hostname in {"localhost", "127.0.0.1"} else "https"
        base_url: str = f"{protocol}://{self.hostname}:{self.port}"
        self.url: str = f"{base_url}/api/{self.api_version}"
        self.retries: int = retries


class RequestClient(_HTTPClient, ABCRequestClient):
    """Async API client.

    This class is the async implementation of the API client
    """

    async def _request_retry(
        self,
        uri: str,
        method: HttpMethod,
        data: dict | aiohttp.FormData | None = None,
    ) -> dict | list[dict]:
        """Execute the async get "aiohttp" by adding class-level settings.

        Parameters
        ----------
        uri: str
            Full http url
        data: dict | aiohttp.FormData
            in case of post, the posted data, empty dict otherwise
        method: HttpMethod
            request type enum

        Returns
        -------
        FIXME.Response object
            will raise_for_status so caller
            may use .json()

        """
        # URLEncode the uri
        uri = urllib3.util.parse_url(uri).url

        random.seed()
        sleep_for: int = random.randrange(1, 5)
        exception: aiohttp.client_exceptions.ClientResponseError | ValueError

        is_data = isinstance(data, aiohttp.FormData)
        is_json = isinstance(data, dict)

        logger.debug(f"Requesting {uri} with {method} method")
        logger.debug(f"is_data: {is_data}")
        logger.debug(f"is_json: {is_json}")

        for _ in range(self.retries):
            try:
                async with aiohttp.ClientSession(
                    headers=self.headers, raise_for_status=True
                ) as session:
                    resp: aiohttp.ClientResponse

                    if is_data:
                        resp = await session.request(
                            method.value,
                            uri,
                            data=data,
                        )

                    elif is_json:
                        resp = await session.request(
                            method.value,
                            uri,
                            json=data,
                        )

                    elif data is None and method == HttpMethod.get:
                        resp = await session.request(
                            method.value,
                            uri,
                        )

                    else:
                        raise ValueError(f"Invalid Content Type found: {type(data)}")

                    retval: dict | list[dict] = await resp.json()

                    return retval

            except aiohttp.client_exceptions.ClientResponseError as e:
                await asyncio.sleep(sleep_for)
                sleep_for = random.randrange(sleep_for, 2 * sleep_for)
                exception = e
            except ValueError as e:
                exception = e
                break

        raise exception

    async def get(self, endpoint: str) -> dict:
        """Execute a get request.

        Convenience wrapper around aiohttp.get (called via _get method)

        Parameters
        ----------
        endpoint: str
            a valid prefix to the orchestration API (including guid
            if applicable) and will add to the dns prefix

        Returns
        -------
            dict: unpacked json dict

        """
        response_data = await self._request_retry(
            f"{self.url}/{endpoint}",
            HttpMethod.get,
        )

        if not isinstance(response_data, dict):
            raise RuntimeError("Expected a single object, got a list")

        return response_data

    async def get_many(self, endpoint: str) -> list[dict]:
        """Execute a get request.

        Convenience wrapper around aiohttp.get (called via _get method)

        Parameters
        ----------
        endpoint: str
            a valid prefix to the orchestration API (including guid
            if applicable) and will add to the dns prefix

        Returns
        -------
            list[dict]: unpacked json dict

        """
        response_data = await self._request_retry(
            f"{self.url}/{endpoint}",
            HttpMethod.get,
        )

        if not isinstance(response_data, list):
            raise RuntimeError("Expected a list of objects, got a single object")

        return response_data

    async def post(
        self,
        endpoint: str,
        data: dict | aiohttp.FormData,
    ) -> dict:
        """Execute a post request.

        Convenience wrapper around requests.post (called via _post method)

        Parameters
        ----------
        endpoint: str
            a valid prefix to the orchestration API (including guid
            if applicable) and will add to the dns prefix
        data: dict
            json-able data payload
        content_type: Literal['json'] | Literal['data']
            content type of the post request. For `data` content type,
            a FormData object is expected in the `data` parameter

        Returns
        -------
            dict: unpacked json dict

        """
        response_data = await self._request_retry(
            f"{self.url}/{endpoint}", HttpMethod.post, data
        )

        if not isinstance(response_data, dict):
            raise RuntimeError("Expected a single object, got a list")

        return response_data
