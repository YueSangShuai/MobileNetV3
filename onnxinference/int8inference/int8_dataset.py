from PIL import Image
from openvino.tools.pot import Metric, DataLoader, IEEngine
import os
from torchvision import transforms
from cv2 import imread, resize as cv2_resize
import torch


class ImageNetDataLoader(DataLoader):

    def __init__(self, config,transform=None):
        super().__init__(config)
        
        self.data_info = self.get_img_info(config.get('data_source'))
        self.transform = transform
        
        
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        if torch.is_tensor(img):
            img=img.numpy()
        
        return img, None

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # print(dirs)
            # 遍历类别
            for sub_dir in dirs:
                # listdir为列出文件夹下所有文件和文件夹名
                img_names = os.listdir(os.path.join(root, sub_dir))
                # 过滤出所有后缀名为jpg的文件名（那当然也就把文件夹过滤掉了）
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    # 在该任务中，文件夹名等于标签名
                    label = sub_dir
                    # data_info.append((path_img, dict.get(label)))
                    data_info.append((path_img, int(label)))
        return data_info



if __name__=="__main__":
    dataset_config = {
        'data_source': os.path.expanduser("/mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/train"),
        }
    
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
    
    dataloader=ImageNetDataLoader(dataset_config,data_transform['val'])
    print(dataloader[0][0].shape)
