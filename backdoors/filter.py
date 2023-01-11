import numpy as np
import pilgram
import torch
from PIL import Image

class Filter:
    def __init__(self):
        pass

    def inject(self, inputs):
        inputs = inputs[0].permute((1, 2, 0)).numpy()
        inputs = (inputs * 255.0).astype(np.uint8)
        inputs = Image.fromarray(inputs)

        inputs = pilgram.nashville(inputs)

        inputs = np.array(inputs) / 255.0
        inputs = torch.Tensor(inputs).permute((2, 0, 1)).unsqueeze(0)
        inputs = torch.clamp(inputs, 0.0, 1.0)
        return inputs
