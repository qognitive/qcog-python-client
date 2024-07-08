import os
from subprocess import Popen
import requests

def pull_openapi_schema() -> dict:
    """This function is meant to pull the OpenAPI json schema from a source.1
    For now the source is a live endpoint, eventually could be pointed to a S3 bucket.
    """

    SOURCE = "http://127.0.0.1:8000/openapi.json"
    OUT = "schema.json"
    response = requests.get(SOURCE)

    with open(OUT, "w") as f:
        f.write(response.text)

    return OUT


def main():
    schema_address = pull_openapi_schema()

    CODE_GEN_OUT = "qcog_python_client/schema/generated_schema/models.py"

    # Create the folder if it doesn't exist
    if not os.path.exists(CODE_GEN_OUT):
        os.makedirs(os.path.dirname(CODE_GEN_OUT), exist_ok=True)

    # Generate pydantic models out of the schema
    r = Popen([
        "datamodel-codegen",
        "--input", schema_address,
        "--output", CODE_GEN_OUT,
        "--use-schema-description",
        "--output-model-type", "pydantic_v2.BaseModel",
    ])
    r.wait()


if __name__ == "__main__":
    main()