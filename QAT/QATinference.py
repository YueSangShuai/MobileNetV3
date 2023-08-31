import torch
import torchvision.transforms as transforms
from PIL import Image
from onnxinference.onnxinference import predict_onnx, get_onnx_model
from onnxinference.openvinoinference import openvioninfer, get_openvino_model
from tqdm import tqdm
import time
from model.MobileNetV3 import MobileNetV3Config
from Pythonproject.MobileNetV3.QATtrain import QuantizedMobileNetV3


def load_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)
    return model



def QATIngerence(model, pic_path):
    model.eval()
    img = Image.open(pic_path).convert('RGB')
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(img).unsqueeze(0)
    rescult_data = model(img_tensor)

    _, predicted = torch.max(rescult_data.data, 1)
    result = predicted[0].item()
    # print("预测的结果为：", rescult_data)

    return rescult_data


if __name__ == "__main__":
    device = torch.device("cpu")
    model_yaml = "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp5/QAT/best.pt"
    model=load_model(model_yaml,device)
    # QATIngerence