import torch.nn as nn
from config import ACTIVATE_FUNC


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        self.first_conv = self.set_first_conv()
        self.depthwise_conv = self.set_depthwise_conv()
        self.separable_conv = self.set_separable_conv()
        self.classify = self.set_classify()

    def forward(self, x):
        outputs = self.first_conv(x)
        outputs = self.depthwise_conv(outputs)
        outputs = self.separable_conv(outputs)
        outputs = outputs.view((outputs.size(0), -1))
        outputs = self.classify(outputs)

        return outputs

    def set_network(self):
        return nn.Sequential(*(self.first_conv + self.depthwise_conv + self.separable_conv + self.classify))

    def set_first_conv(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def set_depthwise_conv(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activate_func,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )

    def set_separable_conv(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activate_func,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )

    def set_classify(self):
        return nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    @property
    def activate_func(self):
        return {
            'ReLU': nn.ReLU(),
            'ELU': nn.ELU(alpha=1.0),
            'LeakyReLU': nn.LeakyReLU(negative_slope=0.01)
        }[ACTIVATE_FUNC]


class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
        self.network = self.set_network()
        self.classify = self.set_classify()

    def forward(self, x):
        outputs = self.network(x)
        outputs = outputs.view((outputs.size(0), -1))
        outputs = self.classify(outputs)

        return outputs

    def set_network(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5), stride=(1, 1), bias=False),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(2, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activate_func,
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 5), stride=(1, 1), bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activate_func,
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 5), stride=(1, 1), bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activate_func,
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 5), stride=(1, 1), bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activate_func,
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.5)
        )

    def set_classify(self):
        return nn.Sequential(
            nn.Linear(in_features=8600, out_features=2, bias=True)
        )

    @property
    def activate_func(self):
        return {
            'ReLU': nn.ReLU(),
            'ELU': nn.ELU(alpha=1.0),
            'LeakyReLU': nn.LeakyReLU(negative_slope=0.01)
        }[ACTIVATE_FUNC]
