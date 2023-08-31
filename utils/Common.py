import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial
from typing import List


# ------------------------------------------------------#
#   这个函数的目的是确保Channel个数能被8整除。
#   离它最近的8的倍数
# 	很多嵌入式设备做优化时都采用这个准则
# ------------------------------------------------------#
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    # int(v + divisor / 2) // divisor * divisor：四舍五入到8
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# -------------------------------------------------------------#
#   Conv+BN+Acti经常会用到，组在一起
# -------------------------------------------------------------#
class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer=None,  # 卷积后的BN层
        activation_layer=None,
    ):  # 激活函数
        padding = (kernel_size - 1) // 2
        if norm_layer is None:  # 没有传入，就默认使用BN
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),  # 后面会用到BN层，故不使用bias
            norm_layer(out_planes),
            activation_layer(inplace=True),
        )


# ------------------------------------------------------#
#   注意力模块：SE模块
# 	就是两个FC层，节点个数、激活函数要注意要注意
# ------------------------------------------------------#
class SqueezeExcitation(nn.Module):
    # squeeze_factor: int = 4：第一个FC层节点个数是输入特征矩阵的1/4
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        # 第一个FC层节点个数，也要是8的整数倍
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        # 通过卷积核大小为1x1的卷积替代FC层，作用相同
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x有很多channel，通过output_size=(1, 1)实现每个channel变成1个数字
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        # 此处的scale就是第二个FC层输出的数据
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x  # 和原输入相乘，得到SE模块的输出


class BottleNeck(nn.Module):
    def __init__(
        self,
        input_c: int,
        kernel: int,
        expanded_c: int,  # bottleneck中的第一层1x1卷积升维，维度升到多少
        out_c: int,
        use_se: bool,
        activation: str,
        stride: int,
        width_multi: float,
        norm_layer=None,
    ):
        super(BottleNeck, self).__init__()
        # 和mobilenetv2中倍率因子相同，通过它得到每一层channels个数和基线的区别
        self.input_c = self.adjust_channels(input_c, width_multi)  # 倍率因子用在这儿了
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        # activation == "HS"，则self.use_hs==True
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

        # 是否使用shortcut连接
        self.use_res_connect = self.stride == 1 and self.input_c == self.out_c
        activation_layer = nn.Hardswish if self.use_hs else nn.ReLU
        self.skip_add = nn.quantized.FloatFunctional()
        layers: List[nn.Module] = []

        # expand
        if self.expanded_c != self.input_c:  # 第一个bottleneck没有这个1x1卷积，故有这个if哦安短
            layers.append(
                ConvBNActivation(
                    self.input_c,
                    self.expanded_c,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvBNActivation(
                self.expanded_c,  # 上一层1x1输出通道数为cnf.expanded_c
                self.expanded_c,
                kernel_size=self.kernel,
                stride=self.stride,
                groups=self.expanded_c,  # DW卷积
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        if self.use_se:  # 是否使用se模块，只需要传入个input_channel
            layers.append(SqueezeExcitation(self.expanded_c))

        # project       降维1x1卷积层
        layers.append(
            ConvBNActivation(
                self.expanded_c,
                self.out_c,
                kernel_size=1,
                norm_layer=norm_layer,
                # nn.Identity是一个线性激活，没进行任何处理
                #    内部实现：直接return input
                activation_layer=nn.Identity,
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = self.out_c

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            self.skip_add.add(result, x)

        return result

    # 静态方法
    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


def adjust_channels(channels: int, width_multi: float):
    return _make_divisible(channels * width_multi, 8)
