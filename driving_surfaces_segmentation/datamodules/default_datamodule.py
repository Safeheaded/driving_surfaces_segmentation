from torch.utils.data import DataLoader
import os
import lightning as L
from ..datasets.default_dataset import DefaultDataset
from ..data_fetchers.RoboflowClient import RoboflowClient
from .helpers import handle_robflow_dataset, handle_google_drive_files
from pathlib import Path
from ..data_fetchers.GoogleDriveClient import GoogleDriveClient
from sklearn.model_selection import train_test_split
import albumentations as A
import albumentations.pytorch.transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class DefaultDatamodule(L.LightningDataModule):
    def __init__(self, data_dir="datasets", batch_size=2, num_workers=4, version = "3", overwrite=False):
        super().__init__()
        print('Init datamodule')
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.version = version
        self.overwrite = overwrite
        self.dataset_location = Path(os.path.join(Path.cwd(), self.data_dir))
        self.google_drive_path = self.dataset_location / "pan_geodeta"
        self.location = self.dataset_location / (os.getenv("ROBOFLOW_PROJECT_NAME") + "--" + self.version)
        self.test_paths = []
        self.test_paths_labels = []

        self.augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.Normalize(mean=0.0, std=1.0),
            A.Resize(height=256, width=256),
            A.pytorch.transforms.ToTensorV2(),
        ])

        self.transforms = A.Compose([
            # A.Normalize(mean=0.0, std=1.0),
            A.Resize(height=256, width=256),
            A.pytorch.transforms.ToTensorV2(),
        ])

    def prepare_data(self):
        # if not self.google_drive_path.exists():
        #     google_drive = GoogleDriveClient()
        #     google_drive.getDataset()
        print(self.location)
        if not os.path.exists(self.location):
            roboflow = RoboflowClient()
            dataset = roboflow.getDataset(self.version)
            dataset_path = Path(dataset.location)
            folders = [dataset_path / Path(item.name) for item in dataset_path.iterdir() if item.is_dir()]
            handle_robflow_dataset(folders)
            # handle_google_drive_files(self.location)


    def setup(self, stage=None):
        # Ścieżki do folderów z danymi
        train_image_dir = self.location / "train/images"
        train_label_dir = self.location / "train/labels"

        all_images = sorted(list(train_image_dir.glob("*.png")))
        all_labels = sorted(list(train_label_dir.glob("*.png")))

        images_train, images_val, labels_train, labels_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
        images_val, images_test, labels_val, labels_test = train_test_split(images_val, labels_val, test_size=0.3, random_state=42)

        self.test_paths_images = images_test
        self.test_paths_labels = labels_test



        # Datasety
        self.train_dataset = DefaultDataset(images_train, labels_train, transform=self.augmentations)
        self.val_dataset = DefaultDataset(images_val, labels_val, transform=self.transforms)
        self.test_dataset = DefaultDataset(images_test, labels_test, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    
    def get_test_paths(self):
        return self.test_paths_images, self.test_paths_labels
