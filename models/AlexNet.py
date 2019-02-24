import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # self.apply(self.init_weight)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    # def init_weight(m):
    #     if type(m) == nn.Linear:
    #         nn.init.xavier_normal_(m.weight)
    #         m.bias.data.fill_(0.01)
    #     elif type(m) == nn.Conv2d:
    #         nn.init.kaiming_normal_(m.weight)
    #         m.bias.data.fill_(0.0)

    # def init_weight(self):
    #     for m in self.modules():
    #         if type(m) == nn.Linear:
    #             nn.init.xavier_normal_(m.weight)
    #             m.bias.data.fill_(0.01)
    #             print(m.weight.device)
    #
    #         elif type(m) == nn.Conv2d:
    #             nn.init.kaiming_normal_(m.weight)
    #             m.bias.data.fill_(0.0)
    #             print(m.weight.device)

