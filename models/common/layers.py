import torch.nn as nn

class GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x, edge_index):
        # Dummy implementation
        return self.linear(x)
