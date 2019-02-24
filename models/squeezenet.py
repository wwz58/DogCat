from collections import OrderedDict

import torch
from torch import nn


class Fire(nn.Module):
    def __init__(self, cin, s1, e1, e3):
        super(Fire, self).__init__()
        self.s1 = nn.Conv2d(cin, s1, 1)
        self.e1 = nn.Conv2d(s1, e1, 1)
        self.e3 = nn.Conv2d(s1, e3, 3, 1, 1)

    def forward(self, x):
        x = self.s1(x)
        x = nn.ReLU(inplace=True)(x)
        y1 = self.e1(x)
        y2 = self.e3(x)
        y = torch.cat([y1, y2], 1)
        y = nn.ReLU(inplace=True)(y)
        return y


class SqueezeNet(nn.Module):
    def __init__(self, num_class=2):
        super(SqueezeNet, self).__init__()
        self.arch = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 96, 7, 2, 2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fire2', Fire(96, 16, 64, 64)),
            ('fire3', Fire(128, 16, 64, 64)),
            ('fire4', Fire(128, 32, 128, 128)),
            ('max_pool4', nn.MaxPool2d(3, 2)),
            ('fire5', Fire(256, 32, 128, 128)),
            ('fire6', Fire(256, 48, 192, 192)),
            ('fire7', Fire(384, 48, 192, 192)),
            ('fire8', Fire(384, 64, 256, 256)),
            ('max_pool8', nn.MaxPool2d(3, 2)),
            ('fire9', Fire(512, 64, 256, 256)),
            ('droupout9', nn.Dropout2d(inplace=True)),
            ('conv10', nn.Conv2d(512, num_class, 1)),
            ('avg_pool10', nn.AvgPool2d(13, 1))
        ]))

    def forward(self, x):
        x = self.arch(x)
        print(x.size())
        assert (x.size()[2:] == (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def init_weight(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
                print(m.weight.device)

            elif type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.0)
                print(m.weight.device)
