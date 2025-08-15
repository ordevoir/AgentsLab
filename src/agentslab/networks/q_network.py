import torch.nn as nn

_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh}

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(128,64), activation="relu"):
        super().__init__()
        act = _ACT.get(activation, nn.ReLU)
        layers=[]
        last=obs_dim
        for h in hidden_sizes: layers += [nn.Linear(last,h), act()] 
        last=h
        layers += [nn.Linear(last, action_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): 
        return self.net(x)
