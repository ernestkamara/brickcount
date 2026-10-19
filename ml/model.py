"""Tiny seam classifier: input [N,3,16,32] RGB in 0..1 (pixel/255), output logit [N,1] (seam > 0)."""
import torch
import torch.nn as nn


def cbr(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SeamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            cbr(3, 16), cbr(16, 24), nn.MaxPool2d(2),     # 8 x 16
            cbr(24, 32), cbr(32, 48), nn.MaxPool2d(2),    # 4 x 8
            cbr(48, 48),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48 * 8, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))

    def forward(self, x):
        x = (x - 0.5) * 4.0               # normalisation baked into the graph; input is pixel/255
        x = self.features(x)
        x = x.mean(dim=2)                 # pool along the gap direction -> [N,48,8] (keeps position across gap)
        return self.head(x)


def param_count(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = SeamNet()
    print(param_count(m), m(torch.zeros(2, 3, 16, 32)).shape)
