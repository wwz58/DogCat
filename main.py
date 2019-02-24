import os

import numpy as np
import pandas as pd
import torch as t
import torchvision.transforms as T
from torch import nn
from torch import optim
from torch.utils import data
from torchnet import meter as m
from tqdm import tqdm

import models
from config import opt
from dataset import DogCat


# import sys

# sys.path.append("pycharm-debug-py3k.egg")
# import pydevd
#
# pydevd.settrace('172.20.208.23', port=12345, stdoutToServer=True,
#                 stderrToServer=True)


# from visualize import Visualize


def train(**kwargs):
    opt.update(kwargs)
    # model
    model = getattr(models, opt.model, 'AlexNet')()
    # model.apply(model.init_weight)
    # model.init_weight()
    model.to(opt.device)

    model.train()
    # data
    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize(256),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = DogCat(opt.train_root, train_ration=opt.train_ration, transform=transform_train)
    val_data = DogCat(opt.train_root, test=False, train_ration=opt.train_ration, transform=transform_val)
    train_loader = data.DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers)
    val_loader = data.DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.num_workers)
    # train
    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = optim.Adam(model.parameters(), lr)
    loss_meter = m.AverageValueMeter()
    train_cm = m.ConfusionMeter(2)
    prev_loss = 1e10
    # vis = Visualize(env='DogCat')
    for epoch in range(opt.max_epoches):
        loss_meter.reset()
        train_cm.reset()
        for ite, (batch, label) in tqdm(enumerate(train_loader)):
            batch, label = batch.to(opt.device), label.to(opt.device)
            # print(batch.device, label.device)
            optimizer.zero_grad()
            # pdb.set_trace()
            # pydevd.settrace()

            score = model.forward(batch)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()
            # record statistics
            loss_meter.add(loss.item())
            train_cm.add(score.detach(), label.detach())
            # plot loss
            if (ite + 1) % opt.plot_every == 0:
                # vis.line(np.asarray([loss_meter.value()[0]]), X=np.asarray([loss_x]),
                #          update='append', win='average_loss')
                # loss_x += 1
                # vis.plot('loss', loss_meter.value()[0])
                print('epoch: %s, lr: %s, loss: %s, train_cm: %s, batch: %s' %
                      (epoch, lr, loss_meter.value()[0], train_cm.value(), ite))

        # save every epoch and logging statistics
        val_cm, val_acc = val(model, val_loader)
        # log = '[{time}] epoch:{epoch}, lr:{lr}, loss:{loss}, \
        #             train_cm:{train_cm}, val_cm:{val_cm}, \
        #             val_acc:{val_acc}' \
        #     .format(time=time.strftime('%m%d_%H%M%S'),
        #             epoch=epoch,
        #             lr=lr,
        #             loss=loss_meter.value()[0],
        #             train_cm=str(train_cm.value()),
        #             val_cm=str(val_cm.value()),
        #             val_acc=val_acc
        #             )
        # log_text += log
        # vis.text(log_text, win='epoch_logging')
        # vis.log('epoch: %s, lr: %s, loss: %s, train_cm: %s, val_cm: %s, val_acc: %5f' %
        #         (epoch, lr, loss_meter.value()[0], train_cm.value(), val_cm.value(), val_acc))

        print('epoch: %s, lr: %s, loss: %s, train_cm: %s, val_cm: %s, val_acc: %5f' %
              (epoch, lr, loss_meter.value()[0], train_cm.value(), val_cm.value(), val_acc))
        model_name = 'model_%02depoch_%.5f.pt' % (epoch, val_acc)
        t.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               os.path.join(opt.model_path, model_name)
               )
        # update lr
        if loss_meter.value()[0] > prev_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        prev_loss = loss_meter.value()[0]


@t.no_grad()
def val(model, val_loader):
    val_cm = m.ConfusionMeter(2)
    model.eval()
    for batch, label in tqdm(val_loader):
        batch, label = batch.to(opt.device), label.to(opt.device)
        score = model.forward(batch)
        val_cm.add(score.detach(), label.detach())
    val_acc = np.trace(val_cm.value()) / np.sum(val_cm.value())
    return val_cm, float(val_acc)


@t.no_grad()
def test(**kwargs):
    opt.update(kwargs)
    model = getattr(models, opt.model, 'AlexNet')()
    model.load_state_dict(t.load(opt.load_model, map_location=opt.device)['model_state_dict'])
    model.to(opt.device)
    model.eval()
    transform_val = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = DogCat(opt.test_root, test=True, transform=transform_val)
    data_loader = data.DataLoader(dataset, batch_size=128, num_workers=opt.num_workers)
    result = []
    for batch, path in tqdm(data_loader):
        batch = batch.to(opt.device)
        score = model.forward(batch)
        predict = score.max(dim=1)[1].detach().tolist()
        result += [(path_, predict_) for path_, predict_ in zip(path, predict)]
    result = pd.DataFrame(result)
    result.to_csv(opt.result_file)


def helper():
    print('''
    usage:
    python3 main.py train --env="DogCat" --train_root="data\\train" --lr=1e-2
    python3 main.py test --test_root="data\\train"
    python3 main.py helper
    ''')


if __name__ == '__main__':
    import fire

    fire.Fire()
    train()  # , model='SqueezeNet'
    # test(load_model='models\\model_00epoch_0.00000.pt')
