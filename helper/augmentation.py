import kornia.augmentation as A
import random
import torch

class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    def __init__(self, shape):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(A.RandomCrop(shape, padding=5), p=0.8)
        self.random_rotation = ProbTransform(A.RandomRotation(10), p=0.5)
        self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
