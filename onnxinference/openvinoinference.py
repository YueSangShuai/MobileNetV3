import openvino.runtime as ov
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


def get_openvino_model(xml_path):
    core = ov.Core()
    model = core.read_model(xml_path)

    compiled_model = core.compile_model(model, "CPU")

    infer_request = compiled_model.create_infer_request()

    return infer_request


def openvioninfer(infer_request, pic_path):
    # 1、创建openvino初始化的引擎

    from PIL import Image

    img = Image.open(pic_path).convert('RGB')
    transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ]
    )

    img_tensor = transform(img).unsqueeze(0)
    res = infer_request.infer(inputs={"input": img_tensor})

    output_tensor = infer_request.get_output_tensor()
    temp = output_tensor.data[0]
    print(type(output_tensor))
    # print(temp)
    # print("最大的下标为:", np.argmax(temp))
    return output_tensor


if __name__ == "__main__":
    xml_path = "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/bestfp16.xml"
    ir_model = get_openvino_model(xml_path)
    img_path = (
        "/mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/val/0/21652746_cc379e0eea_m.jpg"
    )
    print(openvioninfer(ir_model, img_path))
