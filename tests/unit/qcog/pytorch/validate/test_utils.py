import io

from qcog_python_client.qcog.pytorch.validate.utils import get_third_party_imports


def test_get_third_party_imports_no_imports():
    source_code = io.BytesIO(b"")
    result = get_third_party_imports(source_code)
    assert result == set()


def test_get_third_party_imports_standard_library_imports():
    # asyncio and os are system dependent modules
    # and are more tricky to test cause they are
    # in a specific location
    source_code = io.BytesIO(b"import os\nimport sys\nimport asyncio")
    result = get_third_party_imports(source_code)
    assert result == set()


def test_get_third_party_imports_third_party_imports():
    source_code = io.BytesIO(b"import requests\nimport numpy")
    result = get_third_party_imports(source_code)
    assert result == {"requests", "numpy"}


def test_get_third_party_imports_mixed_imports():
    source_code = io.BytesIO(b"import os\nimport requests\nfrom numpy import array")
    result = get_third_party_imports(source_code)
    assert result == {"requests", "numpy"}


def test_get_third_party_imports_with_aliases():
    source_code = io.BytesIO(b"import os as os_module\nimport requests as req")
    result = get_third_party_imports(source_code)
    assert result == {"requests"}


def test_get_third_party_imports_with_module_imports():
    source_code = io.BytesIO(b"from os import path\nfrom requests import get")
    result = get_third_party_imports(source_code)
    assert result == {"requests"}
