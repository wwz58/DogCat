import torch
import warnings
import logging
import os

logging.basicConfig(level=logging.INFO)


class Config(object):
    # train
    model = 'AlexNet'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'models/*.pt'
    max_epoches = 10
    plot_every = 100
    lr = 1e-3
    lr_decay = 0.5
    env = 'DogCat'
    model_path = 'models'
    # test
    load_model = None
    # data
    train_root = os.path.join('data', 'train')
    test_root = os.path.join('data', 'test1')
    train_ration = 0.9
    num_workers = 8
    result_file = 'submission.csv'
    batch_size = 128

    def update(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: not has attr")
            else:
                setattr(self, k, v)
                logging.info('update opt.%s to %s' % (k, v))


opt = Config()
