import argparse
import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import yaml
from model.MobileNetV3 import MobileNetV3Config
import matplotlib.pyplot as plt
from utils.general import one_cycle
from utils.dataset import MyDataset
import math


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def main(args):
    with open(agrs.hyp, 'r', encoding='utf-8') as f:
        hyp = yaml.load(f.read(), Loader=yaml.FullLoader)

    if agrs.device == 'cpu' or agrs.device == '':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + agrs.device)

    print("using {} device.".format(device))

    with open(agrs.data, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    model_weight_path = agrs.weights

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

    # 创建保存文件路径
    root = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(root, 'runs')):
        os.mkdir(os.path.join(root, 'runs'))

    if os.path.exists(os.path.join(root, 'runs', 'train')) == False:
        os.mkdir(os.path.join(root, 'runs', 'train'))

    dir = os.listdir(os.path.join(root, 'runs', 'train'))
    if len(dir) == 0:
        os.mkdir(os.path.join(root, 'runs', 'train', 'exp1'))
        save_path = os.path.join(root, 'runs', 'train', 'exp1')
    else:
        dir_num = [
            int(num[3:]) for num in os.listdir(os.path.join(root, 'runs', 'train'))
        ]
        os.mkdir(os.path.join(root, 'runs', 'train', 'exp' + str(max(dir_num) + 1)))
        save_path = os.path.join(root, 'runs', 'train', 'exp' + str(max(dir_num) + 1))

    train_dataset = MyDataset(
        data_dir=result['train'], transform=data_transform["train"]
    )

    train_num = len(train_dataset)

    nw = min(
        [os.cpu_count(), agrs.batch_size if agrs.batch_size > 1 else 0, 8]
    )  # number of workers
    if nw < agrs.workers:
        nw = agrs.workers
    print('Using {} dataloader workers every process'.format(nw))

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

    # create model
    net = MobileNetV3Config(agrs.cfg)
    if agrs.weights != '':
        # load pretrain weights
        # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
        assert os.path.exists(model_weight_path), "file {} dose not exist.".format(
            model_weight_path
        )
        pre_weights = torch.load(model_weight_path, map_location='cpu')

        # delete classifier weights
        pre_dict = {
            k: v
            for k, v in pre_weights.items()
            if net.state_dict()[k].numel() == v.numel()
        }
        missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    if agrs.freeze:  # 冻结训练:
        # freeze features weights
        for param in net.features.parameters():
            param.requires_grad = False

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer=optim.SGD(params,lr=hyp['lr0'],momentum=hyp['momentum'],weight_decay=hyp['weight_decay'])

    lf = one_cycle(1, hyp['lrf'], agrs.epochs)

    best_acc = 0.0
    train_steps = len(train_loader)

    lr = []
    for epoch in range(agrs.epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        if epoch <= hyp['warmup_epochs']:
            warmup_percent_done = epoch / hyp['warmup_epochs']
            warmup_learning_rate = hyp['warmup_bias_lr'] * warmup_percent_done
            learning_rate = warmup_learning_rate
            optimizer.param_groups[-1]["lr"] = learning_rate
        else:
            optimizer.param_groups[-1]["lr"] = hyp['lr0'] * lf(epoch)

        for step, data in enumerate(train_bar):
            images, labels = data

            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
                epoch + 1, agrs.epochs, loss
            )

        lr.append(optimizer.param_groups[-1]['lr'])

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                # print(outputs)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, agrs.epochs)
        val_accurate = acc / val_num
        print(
            '[epoch %d] train_loss: %.3f  val_accuracy: %.3f'
            % (epoch + 1, running_loss / train_steps, val_accurate)
        )
        torch.save(net, os.path.join(save_path, 'last.pt'))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net, os.path.join(save_path, 'best.pt'))

    x = [n for n in range(len(lr))]
    plt.plot(x, lr)
    plt.savefig(os.path.join(save_path, 'lr.png'))
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        type=str,
        default='/home/yuesang/Pythonproject/MobileNetV3/cfg/premodelcfg/mobilenet_v3_large.pth',
        help='initial weights path',
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default='/home/yuesang/Pythonproject/MobileNetV3/model/mobileV3large.yaml',
        help='small or large',
    )

    parser.add_argument(
        '--hyp',
        type=str,
        default='/home/yuesang/Pythonproject/MobileNetV3/cfg/hyp/trainconfig/hyp.flower.yaml',
        help='hyp',
    )

    parser.add_argument(
        '--data',
        type=str,
        default='/home/yuesang/Pythonproject/MobileNetV3/cfg/datacfg/flower.yaml',
        help='dataset.yaml path',
    )
    parser.add_argument('--freeze', type=bool, default=False, help='freeze training')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
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
