import torch
import torch.nn
import os
import sys


def export(model_path):
    net = torch.load(
        model_path,
        map_location=torch.device('cpu'),
    )
    net.eval()

    input_names = ['input']
    output_names = ['output']
    x = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        net,
        x,
        os.path.join(
            os.path.split(model_path)[0],
            'best.onnx',
        ),
        input_names=input_names,
        output_names=output_names,
        verbose=False,
        opset_version=8,
    )


if __name__ == "__main__":
    model_path = '/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp3/best.pt'
    export(model_path)
