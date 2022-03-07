import torch
from torch import nn
from i3d_base import InceptionI3d


class I3DCustom(nn.Module):
    def __init__(self, num_classes, in_channels, i3d_load=None, dropout_p=0.5):
        super(I3DCustom, self).__init__()
        self.i3d = InceptionI3d(num_classes=400, spatial_squeeze=True, in_channels=in_channels)
        if i3d_load is not None:
            self.i3d.load_state_dict(torch.load(i3d_load))
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(400, num_classes)

    def forward(self, x):
        y, z = self.i3d(x)
        t = torch.mean(z, dim=2)
        x = self.dropout(t)
        x = self.fc(x)
        return x, y, z, t