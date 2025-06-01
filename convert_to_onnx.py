import torch
import matplotlib.pyplot as plt
from driving_surfaces_segmentation.models.default_model import DefaultSegmentationModel
from driving_surfaces_segmentation.datasets.default_dataset import DefaultDataset
import argparse

def main(input: str, output: str):
    model = DefaultSegmentationModel.load_from_checkpoint(input, num_classes=1, T_MAX=16*344)
    
    model.to('cpu')

    model.load_state_dict(torch.load(input, map_location='cpu')['state_dict'])
    model.eval()

    x = torch.rand(1, 3, 512, 512) # eg. torch.rand([1, 3, 256, 256])
    _ = model(x)

    torch.onnx.export(model,
                x,  # model input
                output,  # where to save the model
                export_params=True,
                opset_version=15,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                              'output': {0: 'batch_size'}})
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the model checkpoint (model.ckpt)")
    parser.add_argument('--onnx_path', type=str, required=True, help="Path to save the ONNX model (model.onnx)")
    args = parser.parse_args()

    input_path = args.ckpt_path
    output_path = args.onnx_path

    main(input_path, output_path)
