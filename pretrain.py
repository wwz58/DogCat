import copy
import time

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import DogCat


def train(model, root, train_ration, batch_size, num_workers, gpu_id, epochs,
          start=time.time()):
    if model == 'SqueezeNet':
        model = models.squeezenet1_1(pretrained=True)
        ch_num = model.classifier[1].in_channels
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(ch_num, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        def forward(x):
            x = model.features(x)
            x = model.classifier(x)
            return x.view(x.size(0), 2)

        model.forward = forward
    elif model == 'AlexNet':
        model = models.alexnet(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    for m in model.features.parameters():
        m.require_grad = False

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_sets = {
        'train': DogCat(root + '/train', train=True, test=False, train_ration=train_ration,
                        transform=data_transforms['train']),
        'val': DogCat(root + '/train', train=False, test=False, train_ration=train_ration,
                      transform=data_transforms['val'])
    }
    data_loaders = {
        'train': DataLoader(data_sets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(data_sets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    phases = ['train', 'val']
    device = gpu_id  # 'cuda:' + gpu_id if torch.cuda.is_available() else 'cpu'
    best_val = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, 7)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in phases:
            if phase == 'val':
                model.eval()
            else:
                model.train()
                scheduler.step()

            running_loss = 0.0
            running_corrects = 0

            for batch, y in data_loaders[phase]:

                batch, y = batch.to(device), y.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    score = model(batch)
                    loss = criterion(score, y)

                    _, predicted = torch.max(score.detach(), 1)
                    # predicted.astype_(y)
                    running_corrects += (predicted == y.detach()).sum().item()
                    running_loss += loss.detach().item() * batch_size

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_acc = running_corrects / len(data_sets[phase])
            print('{}: loss {:.4f}, acc {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                if epoch_acc > best_val:
                    best_val = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - start
    print('Traing Complete, using {}m {}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc:{}'.format(best_val))
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    import fire

    fire.Fire()
    train(model='SqueezeNet', root='data', train_ration=0.9, batch_size=128, num_workers=5, gpu_id='cuda:0', epochs=2)
