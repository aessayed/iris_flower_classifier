import torch
import torch.nn as nn

class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 9),
            nn.ReLU(),
            nn.Linear(9, 3)
        )

    def forward(self, x):
        return self.net(x)

def load_model(path='model.pth'):
    model = IrisModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
