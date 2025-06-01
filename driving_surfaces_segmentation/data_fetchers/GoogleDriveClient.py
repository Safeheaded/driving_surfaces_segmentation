import gdown
from pathlib import Path
import os
import zipfile

class GoogleDriveClient:
    def __init__(self, file_id: str = os.getenv("GOOGLE_DRIVE_FILE_ID")):
        self.file_id = file_id

    def getDataset(self):
        url = f'https://drive.google.com/uc?id={self.file_id}'

        path = Path(os.path.join(os.getcwd(), "datasets"))

        if not path.exists():
            path.mkdir()

        output = path / 'pan_geodeta.zip'
        gdown.download(url, str(output), quiet=False)
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(path)
        os.remove(output)