import copy
import importlib
import inspect
import os
import sys
from typing import Any

from pydantic import BaseModel

from qcog_python_client.qcog.pytorch.handler import Handler
from qcog_python_client.qcog.pytorch.types import (
    Directory,
    QFile,
    ValidateCommand,
)
from qcog_python_client.qcog.pytorch.validate.utils import (
    get_third_party_imports,
    is_package_module,
)

# whitelist of allowed modules
default_allowed_modules = {
    "torch",
    "pandas",
    "numpy",
    "sklearn",
    "torchvision",
}


class TrainFnAnnotation(BaseModel):
    """Train function annotation."""

    arg_name: str
    arg_type: Any


def validate_model_module(
    self: Handler[ValidateCommand],
    file: QFile,
    directory: Directory,
    allowed_modules: set[str] | None = None,
) -> Directory:
    """Validate the model module."""
    allowed_modules = allowed_modules or default_allowed_modules
    dir_path = os.path.dirname(file.path)

    # Very naive way to inspect all the package.
    # Assumes one level deep and doesn't recurse.
    modules_found = set()
    for item_path, item in directory.items():
        if item_path.endswith(".py"):
            third_party_modules = get_third_party_imports(item.content, dir_path)
            for module_name in third_party_modules:
                # If the module_name name is the package, skip it
                if module_name == file.pkg_name:
                    continue
                # If the module is contained is not in the package
                if not is_package_module(item_path):
                    modules_found.add(module_name)

    # Check if the modules found are allowed
    if not_allowed := modules_found - allowed_modules:
        raise ValueError(
            f"Found modules not allowed: {not_allowed} or imported outside the package."
        )

    # Check that the model module contains a train function
    module_name = os.path.basename(file.path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file.path)

    if not spec:
        raise ValueError("Model module not found.")

    # Set the path
    sys.path.append(dir_path)

    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise ValueError("Model module loader not found.")

    spec.loader.exec_module(module)

    # Check for the `train` function
    inspected = inspect.getmembers(
        module, lambda x: inspect.isfunction(x) and x.__name__ == "train"
    )

    if not inspected:
        raise ValueError("Model module does not contain a train function.")

    # Inspected returns a list of tuples, where the first element is the name
    # of the function and the second element is the function itself. We expect
    # only one function with the name `train`.
    train_fn = inspected[0][1]
    train_fn_annotations: dict[str, TrainFnAnnotation] = {}

    for ann in train_fn.__annotations__:
        train_fn_annotations[ann] = TrainFnAnnotation(
            arg_name=ann, arg_type=train_fn.__annotations__[ann]
        )

    # Set the train function annotations on the handler
    setattr(self, "train_fn_annotations", train_fn_annotations)

    # Directory has been validated
    return copy.deepcopy(directory)
