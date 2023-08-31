import torch
import torch.nn
import os
import sys
from QATinference import QATIngerence, load_torchscript_model


def export(model_path):
    device = torch.device("cpu")
    net = load_torchscript_model(model_path, device)
    net.eval()

    input_names = ['input']
    output_names = ['output']
    x = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        net,
        x,
        os.path.join(
            os.path.split(model_path)[0],
            'bestQAT.onnx',
        ),
        input_names=input_names,
        output_names=output_names,
        verbose=False,
        opset_version=8,
    )


if __name__ == "__main__":
    model_path = '/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/QAT/best.pt'
    export(model_path)
