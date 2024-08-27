import pandas as pd
import pytest

from qcog_python_client import AsyncQcogClient
from qcog_python_client.schema.generated_schema.models import TrainingStatus


@pytest.mark.asyncio
async def test_upload_pytorch_folder(get_client):
    client: AsyncQcogClient = await get_client()
    client = await client.pytorch(
        model_name="test_model_00", model_path="tests/pytorch_model"
    )

    assert client.pytorch_model is not None
    assert "guid" in client.pytorch_model
    assert client.pytorch_model["model_name"] == "test_model_00"
    assert client.pytorch_model["experiment_name"].startswith("training-pytorch")


@pytest.mark.asyncio
async def test_upload_data(get_client):
    client: AsyncQcogClient = await get_client()
    df_data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    client = await client.data(df_data)

    assert client.dataset is not None
    assert "guid" in client.dataset
    assert client.dataset["format"] == "dataframe"
    assert "project_guid" in client.dataset


@pytest.mark.asyncio
async def test_reload_data(get_client):
    client: AsyncQcogClient = await get_client()
    client: AsyncQcogClient = await get_client()
    df_data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    client = await client.data(df_data)
    dataset_guid = client.dataset["guid"]
    client.preloaded_data(dataset_guid)
    client.dataset["guid"] == dataset_guid


@pytest.mark.asyncio
async def test_train_model(get_client):
    # Full flow test
    client: AsyncQcogClient = await get_client()
    client = await client.pytorch(
        model_name="test_model_01", model_path="tests/pytorch_model"
    )
    df_data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    client = await client.data(df_data)
    client = await client.train_pytorch(
        {
            "epochs": 1,
            "batch_size": 1,
        }
    )

    await client.wait_for_training()
    status = await client.status()

    assert status == TrainingStatus.completed

    assert client.trained_model is not None
    assert client.trained_model["pytorch_model_guid"] == client.pytorch_model["guid"]
    assert "guid" in client.trained_model
    assert client.trained_model["dataset_guid"] == client.dataset["guid"]
    assert (
        client.trained_model["training_parameters_guid"]
        == client.training_parameters["guid"]
    )

    assert "status" in client.trained_model


@pytest.mark.asyncio
async def test_exclusivity_of_methods(get_client):
    client: AsyncQcogClient = await get_client()
    await client.pytorch(model_name="test_model_01", model_path="tests/pytorch_model")
    with pytest.raises(ValueError):
        # Train is not allowed after pytorch
        await client.train(
            5,
            10,
            weight_optimization={},
            get_states_extra={},
        )

    with pytest.raises(AttributeError):
        # No dataset
        await client.train_pytorch(
            {
                "epochs": 1,
                "batch_size": 1,
            }
        )
