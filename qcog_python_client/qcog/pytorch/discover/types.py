"""Package related types."""

from typing import Any, Callable, Coroutine, TypeAlias

from qcog_python_client.qcog.pytorch.handler import Handler
from qcog_python_client.qcog.pytorch.types import DiscoverCommand, QFile

MaybeIsRelevantFile: TypeAlias = Callable[
    [Handler[DiscoverCommand], QFile], Coroutine[Any, Any, QFile | None]
]
