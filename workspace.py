# Extend the torch.Module class to create your own neural network
class LinearNetwork(nn.Module):
    def __init__(self, dataset):
        super(LinearNetwork, self).__init__()
        x, y = dataset[0]
        c, h, w = x.size()  # C is channels, h
        out = y.size(0)
        self.net = nn.Sequential(nn.Linear(c * h * w, out))

    def forward(self, x):
        n, c, h, w = x.size()
        flattened = x.view(n, c * h * w)
        self.net(flattened)