from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(output_dim),
        )

        if input_dim == output_dim:
            self.identity = nn.Identity()
        else:
            self.identity = nn.Conv2d(input_dim, output_dim, kernel_size = 1)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        idnetity = self.identity(x)
        out = self.layer(x)
        out += idnetity
        out = self.ReLU(out)

        return out