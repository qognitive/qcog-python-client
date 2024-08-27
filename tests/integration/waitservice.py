import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def await_service(url: str):
    try:
        retry = Retry(
            total=5,
            backoff_factor=2,
        )

        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("http://", adapter)
        r = session.get(url, timeout=180)
        print("----> Attempting to connect to service <----")
        print("Status code: ", r.status_code)
        r.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        print(f"Fatal Connection Error: {e}")
        raise e
