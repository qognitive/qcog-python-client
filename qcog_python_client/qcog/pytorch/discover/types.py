"""Package related types."""

from typing import Any, Callable, Coroutine, TypeAlias

from qcog_python_client.qcog.pytorch.handler import Handler
from qcog_python_client.qcog.pytorch.types import DiscoverCommand, QFile

IsRelevantFile: TypeAlias = Callable[
    [Handler[DiscoverCommand], QFile], Coroutine[Any, Any, bool]
]
