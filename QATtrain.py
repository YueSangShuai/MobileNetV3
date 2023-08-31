import argparse
import copy
from torch import nn
import torch
import sys
import os
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

from utils.dataset import MyDataset
from torchvision import transforms
import yaml


class QuantizedMobileNetV3(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedMobileNetV3, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def train(agrs):
    import torch.backends.cudnn as cudnn
    
    
    if agrs.device == 'cpu' or agrs.device == '':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + agrs.device)

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

   # ----------------------定义模型-------------------------------

    net = torch.load(agrs.model_path) 
    fused_model = copy.deepcopy(net)

    net.train()
    fused_model.train()

    for module_name, module in fused_model.named_children():
        if "features" in module_name:
            for basic_block_name, basic_block in module.named_children():
                if basic_block.__class__.__name__ == "ConvBNActivation":
                    torch.ao.quantization.fuse_modules_qat(
                        basic_block, [["0", "1"]], inplace=True
                    )
                else:
                    for sub_block_name, sub_block in basic_block.named_children():
                        for third_name, third_block in sub_block.named_children():
                            if third_block.__class__.__name__ == "ConvBNActivation":
                                torch.ao.quantization.fuse_modules_qat(
                                    third_block, [["0", "1"]], inplace=True
                                )

    quantization_net = QuantizedMobileNetV3(fused_model)

    quantization_net.qconfig = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), 
                                                                                weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
    quantization_net = torch.quantization.prepare_qat(quantization_net, inplace=True)

    # print(quantization_net)

    loss_function = nn.CrossEntropyLoss()
    quantization_net.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(
        quantization_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1, last_epoch=-1
    )

    best_acc = 0.0
    train_steps = len(train_loader)

    lr = []

    path_list = agrs.model_path.split('/')[:-1]
    head = "/"
    for path in path_list:
        head = os.path.join(head, path)
    head = os.path.join(head, "QAT")
    print(head)
    if os.path.exists(head) == False:
        os.mkdir(head)


    for epoch in range(agrs.epochs):
        # train
        quantization_net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data

            optimizer.zero_grad()
            logits = quantization_net(images.to(device))
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

        val_accurate = eval(
            quantization_net, validate_loader, device, epoch, agrs.epochs
        )

        print(
            '[epoch %d] train_loss: %.3f  val_accuracy: %.3f'
            % (epoch + 1, running_loss / train_steps, val_accurate)
        )
        
        
        save_model = torch.quantization.convert(quantization_net, inplace=True)
        save_model.eval()
        
        torch.save(save_model.state_dict(),os.path.join(head,"last.pt"))
        
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(save_model.state_dict(),os.path.join(head,"best.pt"))




def eval(quantization_net, validate_loader, device, epoch, epochs):
    quantization_net.eval()
    val_num = len(validate_loader.dataset)
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = quantization_net(val_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
    val_accurate = acc / val_num
    return val_accurate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp6/best.pt',
        help='pt path',
    )

    parser.add_argument(
        '--data',
        type=str,
        default='/home/yuesang/Pythonproject/MobileNetV3/cfg/datacfg/flower.yaml',
        help='dataset.yaml path',
    )
    parser.add_argument('--freeze', type=bool, default=False, help='freeze training')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
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
    train(agrs)
