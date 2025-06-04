import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=0)
        return (weights * x).sum(dim=0)
