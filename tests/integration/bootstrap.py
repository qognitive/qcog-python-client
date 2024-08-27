import requests

from qcog_python_client.log import qcoglogger as logger

project = {
    "guid": "ce066e50-551e-4c55-b617-82abbdbae237",
    "bucket_name": "ubiops-qognitive-default",
    "name": "poc-train-simple-model",
    "org": "test-org",
    "api_secret_name": "ubiops-dev-token",
}

token = {
    "guid": "d60a3c35-9e68-457e-bb50-af9b790fe47a",
    "created_ts": "2024-06-27T18:58:28.846Z",
    "expires_ts": "2084-06-27T18:58:28.846Z",
}


def create_token(endpoint: str) -> tuple[dict, dict]:
    """Utiliy method to create a token and a project in the database.

    This method is used to bootstrap the database with a token and a project.
    **NOTE**: The `name` of the project, the `bucket_name` and the `api_secret_name`
    actually matter when executing some operations. The current `bucket_name` points to
    an S3 bucket that can contains garbage data. In case you want to test a model
    training operation or an Inference operation, you should change the `bucket_name`
    to point to a valid S3 bucket. The name of the project and the api_secret_name
    are already suitable for testing purposes.
    """
    try:
        create_project_response = requests.post(f"{endpoint}/project", json=project)
        project_data = create_project_response.json()
        project_guid = project_data["guid"]

        token.update({"project_guid": project_guid})
        create_token_response = requests.post(f"{endpoint}/token", json=token)
        token_data = create_token_response.json()

        logger.info(
            f"Created Project + Token, \nProject: {project_data},\nToken: {token_data}"
        )
        return token_data, project_data

    except Exception:
        logger.exception("Critical failure in token creation")
        raise
