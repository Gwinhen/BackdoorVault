import torch.nn.functional as F
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(channels, channels, 3, 1, 1),
                        nn.InstanceNorm2d(channels),
                        nn.ReLU(),
                        nn.Conv2d(channels, channels, 3, 1, 1)
                    )
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x) + x))


class CycleGenerator(nn.Module):
    def __init__(self, channels=32, blocks=9):
        super(CycleGenerator, self).__init__()
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(3, channels, 7, 1, 0),
                  nn.InstanceNorm2d(channels),
                  nn.ReLU(True),

                  nn.Conv2d(channels, 2*channels, 3, 2, 1),
                  nn.InstanceNorm2d(2*channels),
                  nn.ReLU(True),

                  nn.Conv2d(2*channels, 4*channels, 3, 2, 1),
                  nn.InstanceNorm2d(4*channels),
                  nn.ReLU(True)]

        for i in range(blocks):
            layers.append(ResBlock(4*channels))

        layers.extend([nn.ConvTranspose2d(4*channels, 4*2*channels, 3, 1, 1),
                       nn.PixelShuffle(2),
                       nn.InstanceNorm2d(2*channels),
                       nn.ReLU(True),

                       nn.ConvTranspose2d(2*channels, 4*channels, 3, 1, 1),
                       nn.PixelShuffle(2),
                       nn.InstanceNorm2d(channels),
                       nn.ReLU(True),

                       nn.ReflectionPad2d(3),
                       nn.Conv2d(channels, 3, 7, 1, 0),
                       nn.Sigmoid()])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class CycleDiscriminator(nn.Module):
    def __init__(self, channels=64):
        super(CycleDiscriminator, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(3, channels, 4, 2, 1, bias=False),
                        nn.LeakyReLU(0.2, inplace=True),

                        nn.Conv2d(channels, 2*channels, 4, 2, 1, bias=False),
                        nn.InstanceNorm2d(2*channels),
                        nn.LeakyReLU(0.2, inplace=True),

                        nn.Conv2d(2*channels, 4*channels, 4, 2, 1, bias=False),
                        nn.InstanceNorm2d(4*channels),
                        nn.LeakyReLU(0.2, inplace=True),

                        nn.Conv2d(4*channels, 8*channels, 4, 1, 1),
                        nn.InstanceNorm2d(8*channels),
                        nn.LeakyReLU(0.2, inplace=True),

                        nn.Conv2d(8*channels, 1, 4, 1, 1),
                        nn.Sigmoid() 
                    )

    def forward(self, x):
        return self.conv(x)
