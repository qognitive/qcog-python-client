from typing import Any, Callable, Coroutine

import pytest

from qcog_python_client import AsyncQcogClient


@pytest.mark.asyncio
async def test_basic_instantiation(
    get_client: Callable[[], Coroutine[Any, Any, AsyncQcogClient]],
):
    client = await get_client()
    assert client.version is not None
