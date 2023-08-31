import torch
from torchvision import transforms
from PIL import Image
import onnxruntime  # to inference ONNX models, we use the ONNX Runtime
import numpy as np


def get_onnx_model(onnx_path):
    session = onnxruntime.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider'],
    )
    return session


def predict_onnx(session, img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(img).unsqueeze(0)

    input_data = img_tensor.numpy()

    # print("input_data shape {}".format(input_data.shape))

    raw_result = session.run([], {'input': input_data})
    # print(raw_result)
    # print("最大的下标为:", np.argmax(np.array(raw_result)))
    return raw_result


if __name__ == '__main__':
    onnx_model = get_onnx_model(
        "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/best.onnx"
    )

    file = "/mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/val/daisy/105806915_a9c13e2106_n.jpg"
    rescult_data = predict_onnx(onnx_model, file)
