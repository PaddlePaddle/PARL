import paddle.fluid as fluid


class CartpoleModel(fluid.dygraph.Layer):
    def __init__(self, name_scope, act_dim):
        super(CartpoleModel, self).__init__(name_scope)
        hid1_size = act_dim * 10
        self.fc1 = fluid.FC('fc1', hid1_size, act='tanh')
        self.fc2 = fluid.FC('fc2', act_dim, act='softmax')

    def forward(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        return out
