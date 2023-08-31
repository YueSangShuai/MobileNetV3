import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from onnxinference.onnxinference import predict_onnx, get_onnx_model
from onnxinference.openvinoinference import openvioninfer, get_openvino_model
from tqdm import tqdm
import time
from utils.dataset import MyDataset
import sys
from inference import Detector
import onnxruntime  # to inference ONNX models, we use the ONNX Runtime
import numpy as np
import cv2
from QATinference import load_torchscript_model, QATIngerence


def val(net_path, val_pathpath):
    device = torch.device("cuda:0")

    net = torch.load(net_path)
    net.eval()
    net.to(device)

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    validate_dataset = MyDataset(data_dir=val_pathpath, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=1, shuffle=False, num_workers=8
    )

    acc = 0.0  # accumulate accurate number / epoch
    val_bar = tqdm(validate_loader, file=sys.stdout)
    for val_data in val_bar:
        val_images, val_labels = val_data
        outputs = net(val_images.to(device))
        # loss = loss_function(outputs, test_labels)
        predict_y = torch.max(outputs, dim=1)[1]
        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
    val_accurate = acc / val_num
    print(val_accurate)


def val_onnx(net_path, val_pathpath):
    session = onnxruntime.InferenceSession(
        net_path,
        providers=['CPUExecutionProvider'],
    )

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    validate_dataset = MyDataset(data_dir=val_pathpath, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=1, shuffle=False, num_workers=8
    )

    acc = 0.0  # accumulate accurate number / epoch
    val_bar = tqdm(validate_loader, file=sys.stdout)
    for val_data in val_bar:
        val_images, val_labels = val_data
        outputs = session.run([], {'input': val_images.numpy()})
        outputs = torch.tensor(np.array(outputs).reshape(-1, 5))
        # loss = loss_function(outputs, test_labels)
        predict_y = torch.max(outputs, dim=1)[1]
        acc += torch.eq(predict_y, val_labels).sum().item()
    val_accurate = acc / val_num
    print(val_accurate)


def val_pt_onnx(pt_path, onnx_path, val_pathpath):
    device = torch.device("cuda:0")
    net = torch.load(pt_path)
    net.eval()
    net.to(device)

    session = onnxruntime.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider'],
    )

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    validate_dataset = MyDataset(data_dir=val_pathpath, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=1, shuffle=False, num_workers=8
    )

    acc = 0.0  # accumulate accurate number / epoch
    val_bar = tqdm(validate_loader, file=sys.stdout)
    for val_data in val_bar:
        val_images, val_labels = val_data
        outputs1 = net(val_images.to(device))
        # loss = loss_function(outputs, test_labels)
        predict_y1 = torch.max(outputs1, dim=1)[1]

        outputs2 = session.run([], {'input': val_images.numpy()})
        outputs2 = torch.tensor(np.array(outputs2).reshape(-1, 5))
        predict_y2 = torch.max(outputs2, dim=1)[1]

        acc += torch.eq(predict_y1, predict_y2.to(device)).sum().item()
    val_accurate = acc / val_num
    print(val_accurate)


def val_QAT(net_path, val_pathpath):
    device = torch.device("cpu")

    net = load_torchscript_model(net_path, device)
    net.eval()

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    validate_dataset = MyDataset(data_dir=val_pathpath, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=1, shuffle=False, num_workers=8
    )

    acc = 0.0  # accumulate accurate number / epoch
    val_bar = tqdm(validate_loader, file=sys.stdout)
    for val_data in val_bar:
        val_images, val_labels = val_data
        outputs = net(val_images.to(device))
        # loss = loss_function(outputs, test_labels)
        predict_y = torch.max(outputs, dim=1)[1]
        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
    val_accurate = acc / val_num
    print(val_accurate)


def get_filelist(dir):
    Filelist = []

    for home, dirs, files in os.walk(dir):
        for filename in files:
            # 文件名列表，包含完整路径

            Filelist.append(os.path.join(home, filename))

            # # 文件名列表，只包含文件名

            # Filelist.append( filename)

    return Filelist

    img = cv2.imread(path, 0)

    # kernel = np.ones((6, 6), np.uint8)
    # kerne2 = np.ones((3, 3), np.uint8)
    # img = cv2.erode(img, kernel)
    # img=cv2.dilate(img, kerne2)
    cv2.imwrite(imwrite_name, img)


dict = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}


if __name__ == "__main__":
    # val(
    #     "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/best.pt",
    #     "/mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/val/",
    # )
    val_onnx(
        "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/QAT/bestQAT.onnx",
        "/mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/val/",
    )
    # val_pt_onnx(
    #     "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/best.pt",
    #     "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/best.onnx",
    #     "/mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/val/",
    # )
    # val_QAT(
    #     "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/QAT/best.pt",
    #     "/mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/val/",
    # )
