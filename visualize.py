import time

import numpy as np
import visdom


class Visualize(object):
    def __init__(self, env='default'):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False)
        self.idx = {}
        self.text = ''

    # plot a point to a win
    def plot(self, win, y):
        x = self.idx.get(win, 0)
        self.vis.line(np.asarray([y]),
                      X=np.asarray([x]),
                      win=win,
                      update=None if x == 0 else 'append')
        self.idx[win] = x + 1

    # plot many points to a win
    def plot_many(self, d):
        for win, y in d.items():
            self.plot(win, y)

    # log to logging win
    def log(self, info, win='log_text'):
        self.text += '[{time}] {info} <br>'.format(
            time=time.strftime('%m_%d %H:%M:%S'),
            info=info
        )
        self.vis.text(self.text, win)
