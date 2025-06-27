
import os
import getpass
import logging
import time

from rest_client import REST_API_Client

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class STT_REST_API_Client(REST_API_Client):

    def __init__(self,
                 url=None,
                 api_ver=None,
                 base=None,
                 user=getpass.getuser()):

        super().__init__(url, api_ver, base, user)


    def check_health(self, max_try=10, try_wait=30):

        url = f"{self.baseurl}/health"

        for i in range(0, max_try):

            status, output = self.request("GET", url)
            if status:
                return True

            print(f"try ({i+1}/{max_try}): SST health check failed: {output}")

            time.sleep(try_wait)

        return False


    def load_model(self, engine, model_name):

        url = f"{self.baseurl}/models/load"

        params = {
            "engine": engine,
            "model_name": model_name
        }

        return self.request("POST", url, params=params, timeout=5*60)


    def transcribe_file(self, file_path, engine, model_name):

        url = f"{self.baseurl}/transcribe/file"

        params = {
            "engine": engine,
            "model_name": model_name
        }

        with open(file_path, "rb") as f:

            files = {"file": (os.path.basename(file_path), f, "audio/wav")}

            return self.request("POST", url, params=params, files=files, timeout=5*60)
