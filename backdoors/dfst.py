import torch
from models.cyclegan import *

class DFST:
    def __init__(self, normalize, device=None):
        self.device = device
        self.normalize = normalize

        self.genr_a2b = CycleGenerator().to(self.device)
        self.genr_b2a = CycleGenerator().to(self.device)
        self.disc_a   = CycleDiscriminator().to(self.device)
        self.disc_b   = CycleDiscriminator().to(self.device)
        self.genr_a2b = torch.nn.DataParallel(self.genr_a2b)
        self.genr_b2a = torch.nn.DataParallel(self.genr_b2a)
        self.disc_a   = torch.nn.DataParallel(self.disc_a)
        self.disc_b   = torch.nn.DataParallel(self.disc_b)

    def inject(self, inputs):
        return self.normalize(self.genr_a2b(inputs))
