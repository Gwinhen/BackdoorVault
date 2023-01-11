import torch

class Other:
    def __init__(self, attack, device=None):
        self.alpha = 0.2

        if attack == 'blend':
            self.pattern = torch.load('data/trigger/other/blend.pt').to(device)
        elif attack == 'sig':
            self.pattern = torch.load('data/trigger/other/sig.pt').to(device)
        elif attack == 'polygon':
            x, y = 4, 4
            size = 4
            color = [1, 1, 0]
            self.alpha   = torch.zeros((1, 32, 32)).to(device)
            self.pattern = torch.zeros((3, 32, 32)).to(device)
            for i in range(len(color)):
                self.alpha[0, x:x+size, y:y+size] = 1
                self.pattern[i, x:x+size, y:y+size] = color[i]

    def inject(self, inputs):
        inputs = self.alpha * self.pattern + (1 - self.alpha) * inputs
        inputs = torch.clamp(inputs, 0.0, 1.0)
        return inputs
