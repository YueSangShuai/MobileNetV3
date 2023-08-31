import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

soop = "/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp6/best.pt"

# print(head)

# net = torch.load("/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/best.pt")
# net.eval()
# device = torch.device("cpu")
# net.to(device)


# for sub_block_name, sub_block in basic_block.named_children():
#     print(sub_block_name)
# for third_name, third_block in sub_block.named_children():
#     print(third_name)
#     if third_name == "ConvBNActivation":
#         print("666")
#         torch.quantization.fuse_modules(
#             third_block, [["0", "1"]], inplace=True
#         )


net=torch.load(soop)
net.cpu()
net.eval()
classifier = nn.Sequential()
net.classifier[3] = classifier


pic_path = "/mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/val/sunflowers/1484598527_579a272f53.jpg"

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

rescult = net(img_tensor)
# print(net)
print(rescult.shape)
