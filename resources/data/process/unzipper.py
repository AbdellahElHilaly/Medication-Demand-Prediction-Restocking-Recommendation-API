import zipfile
import os

class DataUnzipper:
    """
    Class responsible only for extracting zip archives.
    """
    def __init__(self, zip_path: str, extract_to: str):
        self.zip_path = zip_path
        self.extract_to = extract_to

    def extract(self) -> None:
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Zip file not found at: {self.zip_path}")
        
        os.makedirs(self.extract_to, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_to)
