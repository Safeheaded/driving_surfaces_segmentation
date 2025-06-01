from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import matplotlib.pyplot as plt


class DefaultDataset(Dataset):
    def __init__(self, images: list[str], labels: list[str], transform=None):
        self.image_dir = images
        self.label_dir = labels
        self.transform = transform
        self.image_paths = images
        self.label_paths = labels
        # preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Ścieżki do obrazu i etykiety
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Załaduj obraz
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).copy()

        mask = Image.open(label_path)
        mask = np.array(mask).copy()

        # Zastosuj transformacje jeśli są zdefiniowane
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            # image = np.transpose(image, (2, 0, 1))

        return image, mask

    def visualize_item(self, idx):
        """Wyświetla obrazek i maskę dla danego indeksu"""
        image, mask = self.__getitem__(idx)

        # Konwertuj tensory z powrotem do numpy jeśli to konieczne
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            if image.shape[0] == 3:  # Jeśli kanały są pierwsze (C,H,W)
                image = np.transpose(image, (1, 2, 0))
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        # Normalizuj wartości do zakresu [0,1] jeśli to konieczne
        if image.max() > 1:
            image = image / 255.0
        if mask.max() > 1:
            mask = mask / 255.0

        # Utworzenie subplotów
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Wyświetlenie obrazka
        ax1.imshow(image)
        ax1.set_title("Obraz")
        ax1.axis("off")

        # Wyświetlenie maski
        ax2.imshow(mask, cmap="gray")
        ax2.set_title("Maska")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()
