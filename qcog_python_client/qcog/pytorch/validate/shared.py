"""Shared models for the validation service."""

from pydantic import BaseModel


class FileToValidate(BaseModel):
    """File to validate."""

    path: str
    content: bytes
    pkg_name: str
