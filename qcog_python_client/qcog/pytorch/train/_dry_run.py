import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

from qcog_python_client.qcog.pytorch.handler import BoundedCommand, Handler


async def handle_dry_run(self: Handler, payload: BoundedCommand) -> Any:
    """Handle a dry run."""
    print("============ Dry Run ============")
    get_request = self.get_tool("get_request")
    model_name = self.context.get("model_name")
    model_path = self.context.get("model_path")
    dataset_location = self.context.get("dataset_location")

    # Cross check dataset uuid and dataset location
    # This function is the same used in the backend
    # to generate the dataset uuid based on the location
    # def uuid_from_location(location: str) -> str:
    #     return uuid.uuid3(
    #         uuid.NAMESPACE_OID, location
    #     )

    # model_guid: str = payload.model_guid
    dataset_guid: str = payload.dataset_guid

    print("# Dataset GUID:", dataset_guid)

    # if dataset_guid != uuid_from_location(dataset_location):
    #     raise ValueError("Dataset location and dataset guid do not match.")

    training_parameters_guid: str = payload.training_parameters_guid

    if not model_name:
        raise ValueError(
            "Model name not set in the context. This is required for a dry run."
        )

    if not model_path:
        raise ValueError(
            "Model path not set in the context. This is required for a dry run."
        )

    # # Get the model. TODO: Need endpoint
    # model = await get_request(f"pytorch_model/{model_guid}")

    # Get the dataset
    dataset = await get_request(f"dataset/{dataset_guid}")

    # Get the training parameters
    training_parameters = await get_request(
        f"training_parameters/{training_parameters_guid}"
    )

    print("Dry run:")
    print("Dataset:", dataset)
    print("Training parameters:", training_parameters)

    wrapper_abs_path = Path(__file__).parent / "localtrain_pt.py"
    model_abs_path = Path(model_path).resolve()
    tmp_abs_path = Path(os.getcwd()) / "_tmp"

    print("Wrapper path:", wrapper_abs_path)
    print("Model path:", model_abs_path)
    print("Tmp path:", tmp_abs_path)
    # Set a tmp folder where the model will be copied

    try:
        # Create a tmp folder and copy the content of the model
        shutil.copytree(model_abs_path, tmp_abs_path)
        # Copy the wrapper
        shutil.copy(wrapper_abs_path, tmp_abs_path)
        # Add the path to the system
        sys.path.append(str(tmp_abs_path))
        # Import the wrapper
        wrapper_fn: Callable[..., dict] = None
        try:
            from _tmp.localtrain_pt import train as train_fn

            wrapper_fn = train_fn
        except Exception as e:
            print("Error importing wrapper:", e)
            raise e

        result = wrapper_fn(
            training_data=dataset_location,
            parameters=training_parameters,
            context={"run_id": str(uuid.uuid4())},
            base_directory=None,
        )

        return result

    except Exception as e:
        print("Error copying wrapper:", e)
    finally:
        # Remove the tmp folder
        shutil.rmtree(tmp_abs_path)
