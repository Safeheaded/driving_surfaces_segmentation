import torch
import matplotlib.pyplot as plt
from driving_surfaces_segmentation.models.default_model import DefaultSegmentationModel
from driving_surfaces_segmentation.datasets.default_dataset import DefaultDataset

def main():
    model = DefaultSegmentationModel.load_from_checkpoint('./trained_models/best_model.ckpt', num_classes=1, T_MAX=16*112)
    # model.load_state_dict(torch.load(model_path))
    test_dataset = DefaultDataset("datasets/Segmentacja terenów --1/test/images", "datasets/Segmentacja terenów --1/test/labels")
    first_image, first_mask = test_dataset[1]
    with torch.no_grad():
        first_image = first_image.unsqueeze(0)  # Dodaj wymiar batch
        prediction = model(first_image)
        prediction = prediction.squeeze(0).squeeze(0)
        prediction = prediction.sigmoid()
        prediction = (prediction > 0.5).float()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(first_image.squeeze().permute(1, 2, 0).cpu().numpy())

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(first_mask.cpu().numpy(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(prediction.cpu().numpy(), cmap='gray')

    plt.show()

if __name__ == "__main__":
    main()
