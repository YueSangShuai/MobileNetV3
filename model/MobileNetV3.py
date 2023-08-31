import yaml
import sys

sys.path.append('/home/yuesang/Pythonproject/MobileNetV3/utils/')

from Common import *


class MobileNetV3(nn.Module):
    def __init__(
        self,
        last_channel: int,  # 倒数第二层channel个数
        inverted_residual_setting: List = [],  # 参数设置列表，列表里面每个元素类型是上面定义的那个类的形式
        num_classes: int = 1000,  # 需要分类的类别数
        block=None,
        norm_layer=None,
    ):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (
            isinstance(inverted_residual_setting, List)
            and all([isinstance(s, BottleNeck) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[InvertedResidualConfig]"
            )

        if block is None:
            block = BottleNeck

        # 将norm_layer设置为BN
        #   partial()给输入函数BN指定默认参数，简化之后的函数参数量
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []
        # building first layer   就是普通的conv
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(
            ConvBNActivation(
                3,
                firstconv_output_c,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            cnf.norm_layer = norm_layer
            layers.append(cnf)

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c  # small：96->576; Large:160->960
        layers.append(
            ConvBNActivation(
                lastconv_input_c,
                lastconv_output_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_c, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)  # 到这后面不再需要高和宽的维度了
        x = torch.flatten(x, 1)  # 故进行展平处理
        x = self.classifier(x)

        return x


def MobileNetV3Config(path):
    with open(path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    bneck_conf = partial(
        BottleNeck, width_multi=result['width_multi']
    )  # partial()给输入函数指定默认参数

    reduce_divider = 2 if result['reduce_divider'] else 1

    backbone = result['backbone']

    backbone[-1][0] /= reduce_divider
    backbone[-1][2] /= reduce_divider
    backbone[-1][3] /= reduce_divider

    backbone[-2][0] /= reduce_divider
    backbone[-2][2] /= reduce_divider
    backbone[-2][3] /= reduce_divider

    backbone[-3][3] /= reduce_divider

    inverted_residual_setting: List[nn.Module] = [
        bneck_conf(input_c, kernel, expanded_c, out_c, use_se, activation, stride)
        for [input_c, kernel, expanded_c, out_c, use_se, activation, stride] in backbone
    ]
    last_channel = adjust_channels(
        result['last_channel'] // reduce_divider, result['width_multi']
    )

    return MobileNetV3(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        num_classes=result['nc'],
    )


if __name__ == "__main__":
    model = MobileNetV3Config(
        "/home/yuesang/Pythonproject/MobileNetV3/model/mobileV3large.yaml"
    )

    import torch
    from torchvision import models, transforms
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # summary(model.to(device), (3, 224, 224))

    # print(model)
