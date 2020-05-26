#-*- coding: utf-8 -*-

import parl
from parl import layers


class Model(parl.Model):
    def __init__(self, act_dim):
        act_dim = act_dim
        hid1_size = act_dim * 10

        self.fc1 = layers.fc(size=hid1_size, act='tanh')
        self.fc2 = layers.fc(size=act_dim, act='softmax')

    def forward(self, obs): # 可直接用 model = Model(5); model(obs)调用
        out = self.fc1(obs)
        out = self.fc2(out)
        return out
