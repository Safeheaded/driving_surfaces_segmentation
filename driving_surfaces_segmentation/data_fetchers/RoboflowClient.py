import os
from roboflow import Roboflow
import shutil

class RoboflowClient:
    def __init__(self):
        self.api_key = os.getenv("ROBOFLOW_API_KEY")

        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY not set in environment variables")
        
        self.rf = Roboflow(api_key=self.api_key)
        
    def getDataset(self, version: str, overwrite: bool = False):
        project = self.rf.workspace("automatic-and-robotic").project(os.getenv("ROBOFLOW_PROJECT_NAME"))
        location = os.path.join(os.getcwd(), "datasets", project.name + "--" + version)
        dataset = project.version(version).download(
            "coco-segmentation", location=location, overwrite=overwrite)

        return dataset
