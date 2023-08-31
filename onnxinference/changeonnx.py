import argparse
import os

import torch
import onnx
import onnx.helper as helper
import torchvision.transforms as transforms


class Preprocess(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.transform = transforms.Compose(
        #     [
        #         transforms.Resize(224),
        #         # transforms.CenterCrop(224),
        #     ]
        # )

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def forward(self, x):
        # x原本是uint8的先转为float
        x = x.float()
        x = x[..., [2, 1, 0]]  # BGR to RGB
        x = x.permute(0, 3, 1, 2)
        # x = x[..., [2, 1, 0]]
        print(x.shape)
        x = (x / 255.0 - self.mean.reshape((1, 3, 1, 1))) / self.std.reshape(
            (1, 3, 1, 1)
        )

        return x


def getMeronnx():
    pre = Preprocess()
    # 这里输入名字，尽量自定义，后面转trt可控
    torch.onnx.export(
        pre,
        (torch.zeros((1, 224, 224, 3), dtype=torch.float),),
        "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/pre.onnx",
        input_names=["input"],
    )

    pre = onnx.load("/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/pre.onnx")
    model = onnx.load(
        "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/best.onnx"
    )

    # 先把pre模型名字加上前缀
    for n in pre.graph.node:
        if not n.name == "input":
            n.name = f"{n.name}"
            for i in range(len(n.input)):  # 一个节点可能有多个输入
                n.input[i] = f"{n.input[i]}"
            for i in range(len(n.output)):
                n.output[i] = f"{n.output[i]}"

    # 2 修改另一个模型的信息
    # 查看大模型的第一层名字
    for n in model.graph.node:
        if n.name == "/features/features.0/features.0.0/Conv":
            n.input[0] = pre.graph.output[0].name

    for n in pre.graph.node:
        model.graph.node.append(n)

    # 还要将pre的输入信息 NHWC等拷贝到输入
    model.graph.input[0].CopyFrom(pre.graph.input[0])
    # 此时model的输入需要变为 pre的输入 pre/0
    model.graph.input[0].name = pre.graph.input[0].name

    save_path = os.path.join(
        "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1", "merge.onnx"
    )

    onnx.save(model, save_path)
    # os.unlink("./pre.onnx")


if __name__ == '__main__':
    getMeronnx()
