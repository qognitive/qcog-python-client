from pydantic import BaseModel

from qcog_python_client.qcog.qcogasync import AsyncQcogClient  # noqa
from qcog_python_client.qcog.qcogsync import QcogClient  # noqa

# Globally disable Pydantic protected namespaces warning
BaseModel.model_config["protected_namespaces"] = ()
