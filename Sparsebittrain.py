import argparse
import copy
from torch import nn
import torch
import sys
import os
import numpy as np
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

from sparsebit.quantization import QuantModel, parse_qconfig
from utils.dataset import MyDataset
from torchvision import transforms
import yaml
import torch.backends.cudnn as cudnn
import sparsebit

def main(agrs):
    if agrs.device == 'cpu' or agrs.device == '':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + agrs.device)
    print("using {} device.".format(device))

    with open(agrs.data, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # ----------------------定义数据集-------------------------------
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
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    train_dataset = MyDataset(
        data_dir=result['train'], transform=data_transform["train"]
    )
    
    nw = min(
        [os.cpu_count(), agrs.batch_size if agrs.batch_size > 1 else 0, 8]
    )  # number of workers
    if nw < agrs.workers:
        nw = agrs.workers
    print('Using {} dataloader workers every process'.format(nw))
    
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=agrs.batch_size, shuffle=True, num_workers=nw
    )

    validate_dataset = MyDataset(
        data_dir=result['val'], transform=data_transform["val"]
    )
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=agrs.batch_size, shuffle=False, num_workers=nw
    )

    print(
        "using {} images for training, {} images for validation.".format(
            train_num, val_num
        )
    )
    
    # ----------------------定义模型-------------------------------

    net = torch.load(agrs.model_path) 
    qconfig = parse_qconfig(agrs.qconfig)
    cudnn.benchmark = True
    qmodel = QuantModel(net, qconfig).to(device)
    
    qmodel.model.conv1.input_quantizer.set_bit(bit=8)
    qmodel.model.conv1.weight_quantizer.set_bit(bit=8)
    qmodel.model.fc.input_quantizer.set_bit(bit=8)
    qmodel.model.fc.weight_quantizer.set_bit(bit=8)
    
    

    
    print(qmodel)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp6/best.pt',
        help='pt path',
    )

    parser.add_argument(
        '--qconfig',
        type=str,
        default='Pythonproject/MobileNetV3/cfg/hyp/qconfig/qconfig.yaml',
        help='qconfig',
    )
    
    parser.add_argument(
        '--hyp',
        type=str,
        default='/home/yuesang/Pythonproject/MobileNetV3/cfg/hyp/qconfig/QAT.yaml',
        help='hyp for qat',
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='/home/yuesang/Pythonproject/MobileNetV3/cfg/datacfg/flower.yaml',
        help='dataset.yaml path',
    )
    parser.add_argument('--freeze', type=bool, default=False, help='freeze training')
    parser.add_argument('--epochs', type=int, default=1, help='total training epochs')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='total batch size for all GPUs, -1 for autobatch',
    )

    parser.add_argument(
        '--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='max dataloader workers (per RANK in DDP mode)',
    )
    agrs = parser.parse_args()
    

    main(agrs)
    