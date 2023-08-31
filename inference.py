import torch
import torchvision.transforms as transforms
from PIL import Image
from onnxinference.onnxinference import predict_onnx, get_onnx_model
from onnxinference.openvinoinference import openvioninfer, get_openvino_model
from tqdm import tqdm
import time
from QAT.QATinference import QATIngerence, load_model


class Detector(object):
    # netkind为'large'或'small'可以选择加载MobileNetV3_large或MobileNetV3_small
    # 需要事先训练好对应网络的权重
    def __init__(self, path):
        super(Detector, self).__init__()
        self.net = torch.load(path)
        self.net.eval()
        device = torch.device("cpu")
        self.net.to(device)
        # if torch.cuda.is_available():
        #     self.net.cuda()

    # 检测器主体
    def __call__(self, pic_path):
        # 读取图片
        img = Image.open(pic_path).convert('RGB')
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = transform(img).unsqueeze(0)
        # if torch.cuda.is_available():
        #     img_tensor = img_tensor.cuda()
        net_output = self.net(img_tensor)

        _, predicted = torch.max(net_output.data, 1)
        result = predicted[0].item()
        # print("预测的结果为：", result)
        # print(net_output.data)


if __name__ == '__main__':
    image_path = '/mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/val/sunflowers/1484598527_579a272f53.jpg'
    detector = Detector(
        '/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp5/best.pt'
    )

    # onnx_model = get_onnx_model(
    #     "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/best.onnx"
    # )

    # QATonnx_model = get_onnx_model(
    #     "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/QAT/bestQAT.onnx"
    # )

    # device = torch.device("cpu")
    # QATmodel = load_model(
    #     "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp5/QAT/best.pt",
    #     device,
    # )
    # ir_model = get_openvino_model(
    #     "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/bestfp16.xml"
    # )
    count = 1000

    # # pt模型推理
    # pt_star = time.time()

    # for i in tqdm(range(count)):
    #     detector(image_path)
    # pt_end = time.time()
    # print("pt模型推理时间:", int((pt_end - pt_star) * 1000) / count)

    # # onnx模型推理
    # onnx_start = time.time()
    # for i in tqdm(range(count)):
    #     predict_onnx(
    #         onnx_model,
    #         image_path,
    #     )
    # onnx_end = time.time()
    # print("onnx模型推理时间:", int((onnx_end - onnx_start) * 1000) / count)

    # QAT_onnx_start = time.time()
    # for i in tqdm(range(count)):
    #     predict_onnx(
    #         QATonnx_model,
    #         image_path,
    #     )
    # QAT_onnx_end = time.time()
    # print("QATonnx模型推理时间:", int((QAT_onnx_end - QAT_onnx_start) * 1000) / count)

    # QAT模型

    # QAT_start = time.time()
    # for i in tqdm(range(count)):
    #     QATIngerence(
    #         QATmodel,
    #         image_path,
    #     )
    # QAT_end = time.time()
    # print("pt模型推理时间:", int((QAT_end - QAT_start) * 1000) / count)

    # # openvinofp32
    # openvino_start = time.time()

    # for i in tqdm(range(count)):
    #     openvioninfer(
    #         ir_model,
    #         image_path,
    #     )
    # openvino_end = time.time()
    # print("openvino模型推理时间:", int((openvino_end - openvino_start) * 1000) / count)
