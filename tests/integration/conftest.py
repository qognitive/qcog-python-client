import os
from typing import TypedDict

import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network
from testcontainers.postgres import PostgresContainer

from qcog_python_client import AsyncQcogClient
from qcog_python_client.log import qcoglogger as logger
from tests.integration.bootstrap import create_token
from tests.integration.waitservice import await_service

PORT_SPACE_PREFIX = 20000
PROJECT_NAME_PREFIX = "qcog_python_client"
DB_API_QCOG_ENV = "LOCAL_DOCKER"
ORCHESTRATION_API_QCOG_ENV = "LOCAL_DOCKER"


def port(_port: int) -> int:
    """Add portspace prefix to avoid port collision."""
    return _port + PORT_SPACE_PREFIX


def servicename(sn: str) -> str:
    return f"{PROJECT_NAME_PREFIX}-{sn}"


def aws_credentials():
    return {
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_SESSION_TOKEN": os.getenv("AWS_SESSION_TOKEN"),
        "AWS_REGION": os.getenv("AWS_REGION", "us-east-2"),
    }


class Service(TypedDict):
    container_name: str
    exposed_ports: tuple[int, int]
    image_name: str
    env_vars: dict


postgres_service: Service = {
    "container_name": servicename("postgres"),
    "exposed_ports": (5432, port(5432)),
    "image_name": "postgres:latest",
    "env_vars": {
        "DB_PASSWORD": "password",
        "DB_USERNAME": "postgres",
        "DB_NAME": "postgres",
        "DB_PORT": 5432,
    },
}

crud_api_service: Service = {
    "container_name": servicename("crud-api"),
    "exposed_ports": (5000, port(5000)),
    "image_name": "885886606610.dkr.ecr.us-east-2.amazonaws.com/qcog-db-api",
    "env_vars": {
        "QCOG_ENV": DB_API_QCOG_ENV,
        "QCOG_DB_USERNAME": postgres_service["env_vars"]["DB_USERNAME"],
        "QCOG_DB_PASSWORD": postgres_service["env_vars"]["DB_PASSWORD"],
        "QCOG_DB_HOSTNAME": postgres_service["container_name"],
        "QCOG_DB_NAME": postgres_service["env_vars"]["DB_NAME"],
        "QCOG_DB_PORT": postgres_service["exposed_ports"][0],
        **aws_credentials(),
    },
}

orchestration_api: Service = {
    "container_name": servicename("orchestration-api"),
    "exposed_ports": (8000, port(8000)),
    "image_name": "885886606610.dkr.ecr.us-east-2.amazonaws.com/qcog-orchestration-api",
    "env_vars": {
        "QCOG_ENV": ORCHESTRATION_API_QCOG_ENV,
        "QCOG_DB_API_URL": f"http://{crud_api_service['container_name']}:{crud_api_service['exposed_ports'][0]}",
        **aws_credentials(),
    },
}


@pytest.fixture(scope="session", autouse=True)
def docker_network():
    # Spinning up a docker network
    network = Network().create()

    try:
        yield network
    finally:
        network.remove()


@pytest.fixture(scope="session")
def postgres(docker_network: Network):
    logger.info("--- Starting Postgres Container ---")
    pg_container = (
        PostgresContainer(postgres_service["image_name"])
        .with_name(postgres_service["container_name"])
        .with_bind_ports(*postgres_service["exposed_ports"])
        .with_network(docker_network)
        .with_name(postgres_service["container_name"])
    )

    pg_container.password = postgres_service["env_vars"]["DB_PASSWORD"]
    pg_container.username = postgres_service["env_vars"]["DB_USERNAME"]
    pg_container.dbname = postgres_service["env_vars"]["DB_NAME"]
    pg_container.port = postgres_service["env_vars"]["DB_PORT"]

    try:
        pg_container.start()
        logger.info("... Postgres Container Started ...")
        yield pg_container
    finally:
        pg_container.stop()


@pytest.fixture(scope="session")
def crud_api_mock(postgres: PostgresContainer, docker_network: Network):
    logger.info("--- Starting CRUD API Mock Container ---")
    crud_api_container = (
        DockerContainer(crud_api_service["image_name"])
        .with_command("./local_launch.sh")
        .with_bind_ports(*crud_api_service["exposed_ports"])
        .with_network(docker_network)
        .with_name(crud_api_service["container_name"])
    )

    for key, value in crud_api_service["env_vars"].items():
        crud_api_container = crud_api_container.with_env(key, value)

    try:
        url = f"http://localhost:{crud_api_service['exposed_ports'][1]}/status"
        crud_api_container.start()
        await_service(url)
        logger.info("... CRUD API Mock Container Started ...")
        yield crud_api_container
    finally:
        crud_api_container.stop()


@pytest.fixture(scope="session")
def orchestration_api_mock(crud_api_mock: DockerContainer, docker_network):
    logger.info("--- Starting Orchestration API Mock Container ---")
    orchestration_api_container = (
        DockerContainer(orchestration_api["image_name"])
        .with_command("./local_launch.sh")
        .with_bind_ports(*orchestration_api["exposed_ports"])
        .with_network(docker_network)
        .with_name(orchestration_api["container_name"])
    )

    for key, value in orchestration_api["env_vars"].items():
        orchestration_api_container = orchestration_api_container.with_env(key, value)

    try:
        url = f"http://localhost:{orchestration_api['exposed_ports'][1]}/status"
        orchestration_api_container.start()
        await_service(url)
        logger.info("... Orchestration API Mock Container Started ...")
        yield orchestration_api_container
    except Exception as e:
        logger.error(f"Failed to start Orchestration API Mock Container, {e}")
        raise
    finally:
        orchestration_api_container.stop()


@pytest.fixture(scope="session")
def token(orchestration_api_mock):
    url = f"http://localhost:{crud_api_service['exposed_ports'][1]}/api/internal/v1"
    token_, _ = create_token(url)
    return token_


@pytest.fixture(scope="session")
def get_client(token):
    return lambda: AsyncQcogClient.create(
        token=token["guid"],
        hostname="localhost",
        port=orchestration_api["exposed_ports"][1],
    )
