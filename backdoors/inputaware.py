import torch
from models.generator import Generator

class InputAware:
    def __init__(self, normalize, device=None):
        self.device = device
        self.normalize = normalize

        self.net_mask = Generator(out_channels=1).to(self.device)
        self.net_genr = Generator().to(self.device)
        self.net_mask = torch.nn.DataParallel(self.net_mask)
        self.net_genr = torch.nn.DataParallel(self.net_genr)

    def threshold(self, x):
        return torch.nn.Tanh()(x * 20 - 10) / (2 + 1e-7) + 0.5

    def inject(self, inputs, withp=False):
        inputs = inputs.to(self.device)
        mask = self.threshold(self.net_mask(inputs))
        pattern = self.normalize(self.net_genr(inputs))
        inputs = inputs + (pattern - inputs) * mask
        if withp:
            return inputs, pattern
        else:
            return inputs

    def inject_noise(self, inputs1, inputs2, withp=False):
        mask = self.threshold(self.net_mask(inputs2))
        pattern = self.normalize(self.net_genr(inputs2))
        inputs = inputs1 + (pattern - inputs1) * mask
        if withp:
            return inputs, pattern
        else:
            return inputs
