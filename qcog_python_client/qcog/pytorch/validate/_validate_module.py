import importlib
import inspect
import os
import sys
from dataclasses import dataclass

from pydantic import BaseModel

from qcog_python_client.qcog.pytorch.validate.validate_utils import (
    get_third_party_imports,
    is_package_module,
)


class FileToValidate(BaseModel):
    path: str
    content: str
    pkg_name: str


@dataclass
class TrainFnAnnotation:
    arg_name: str
    arg_type: type


@dataclass
class ValidateModelModule:
    train_fn: dict[str, TrainFnAnnotation]


# whitelist of allowed modules
default_allowed_modules = {
    "torch",
    "pandas",
    "numpy",
    "sklearn",
    "torchvision",
}


def validate_model_module(
    file: FileToValidate,
    allowed_modules: set[str] | None = None,
) -> ValidateModelModule:
    """Validate the model module."""
    allowed_modules = allowed_modules or default_allowed_modules
    dir_path = os.path.dirname(file.path)
    content = os.listdir(dir_path)

    # Very naive way to inspect all the package.
    # Assumes one level deep and doesn't recurse.
    modules_found = set()

    for item in content:
        # Inspect each python file and try to find third-party modules
        if item.endswith(".py"):
            third_party_modules = get_third_party_imports(os.path.join(dir_path, item))
            for module_name in third_party_modules:
                # If the module_name name is the package, skip it
                if module_name == file.pkg_name:
                    continue

                # For each module check if it's part of the current package
                # If not, raise an error
                module_path = os.path.join(dir_path, module_name)

                # If the module is contained is not in the package
                if not is_package_module(module_path):
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

    return ValidateModelModule(train_fn=train_fn_annotations)
