import albumentations as A
import albumentations.pytorch.transforms
from driving_surfaces_segmentation.datamodules.default_datamodule import DefaultDatamodule

augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.pytorch.transforms.ToTensorV2(),
        ])


data_module = DefaultDatamodule(batch_size=16)

data_module.setup()

for i in range(10):
    data_module.train_dataset.visualize_item(i)
