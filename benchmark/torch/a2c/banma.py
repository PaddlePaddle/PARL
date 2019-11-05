# from model import ActorCritic

# banma = ActorCritic(3)
# banma.banma_test()

import torch
import torch.nn as nn


class inner_network_1(nn.Module):
    def __init__(self):
        super(inner_network_1, self).__init__()
        self.inner_fc = nn.Linear(20, 1)

    def forward(self, x):
        x = self.inner_fc(x)
        return x


class inner_network_2(nn.Module):
    def __init__(self):
        super(inner_network_2, self).__init__()
        self.inner_fc = nn.Linear(20, 1)

    def forward(self, x):
        x = self.inner_fc(x)
        return x


class outer_network(nn.Module):
    def __init__(self):
        super(outer_network, self).__init__()
        self.fc = nn.Linear(10, 20)
        self.model_dict = {}
        self.model_dict['model1'] = inner_network_1()
        self.model_dict['model2'] = inner_network_2()

    def forward(self, x):
        x = self.fc(x)
        x = self.model_dict['model1'](x)
        return x


net = outer_network()
# params = net.state_dict()
# for key in params.keys():
#     print(key)
input_tensor = torch.rand([
    10,
])
print(net(input_tensor))
