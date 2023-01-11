import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image

class Refool:
    def __init__(self, shape, mode, device=None):
        self.shape = shape
        self.mode = mode
        self.device = device

        trigger_path = 'data/trigger/refool/000066.jpg'
        trigger = read_image(trigger_path) / 255.0

        if self.mode == 'smooth':
            self.init_smooth(trigger)
        elif self.mode == 'ghost':
            self.init_ghost(trigger)

    def init_smooth(self, trigger):
        self.trigger = transforms.Resize(self.shape)(trigger).to(self.device)
        self.weight_t = self.trigger.mean()

    def init_ghost(self, trigger):
        max_image_size = 560
        offset = (0, 0)
        ghost_alpha = 0.65
        self.alpha_i = 0.65

        h, w = self.shape
        scale_ratio = float(max(self.shape)) / float(max_image_size)
        if w > h:
            self.new_shape = (max_image_size, int(round(h / scale_ratio)))
        else:
            self.new_shape = (int(round(w / scale_ratio)), max_image_size)

        t = transforms.Resize(self.new_shape)(trigger)
        t = t.pow(2.2).permute(1, 2, 0).numpy()

        if offset[0] == 0 and offset[1] == 0:
            offset = (np.random.randint(3, 8), np.random.randint(3, 8))

        t_1 = np.lib.pad(t, ((0, offset[0]), (0, offset[1]), (0, 0)),
                         'constant', constant_values=0)
        t_2 = np.lib.pad(t, ((offset[0], 0), (offset[1], 0), (0, 0)),
                         'constant', constant_values=(0, 0))

        if ghost_alpha < 0:
            ga_switch = 1 if np.random.random() > 0.5 else 0
            ghost_alpha = abs(ga_switch - np.random.uniform(0.15, 0.5))

        ghost_t = t_1 * ghost_alpha + t_2 * (1 - ghost_alpha)
        ghost_t = ghost_t[offset[0]: -offset[0], offset[1]: -offset[1], :]
        ghost_t = torch.FloatTensor(ghost_t).to(self.device).permute(2, 0, 1)
        self.ghost_t = transforms.Resize(self.new_shape)(ghost_t)

        if self.alpha_i < 0:
            self.alpha_i = 1. - np.random.uniform(0.05, 0.45)

    def inject(self, inputs):
        assert (len(inputs.shape) == 4), "Inputs shoud have a shape of 4"
        if self.mode == 'smooth':
            inputs = self.inject_smooth(inputs)
        elif self.mode == 'ghost':
            inputs = self.inject_ghost(inputs)
        return inputs

    def inject_smooth(self, inputs):
        out = []
        for img in inputs:
            weight_i = img.mean()
            param_i =      weight_i / (weight_i + self.weight_t)
            param_t = self.weight_t / (weight_i + self.weight_t)
            img = torch.clamp(param_i * img + param_t * self.trigger, 0.0, 1.0)
            out.append(img)
        inputs = torch.stack(out)
        return inputs

    def inject_ghost(self, inputs):
        out = []
        for img in inputs:
            img = transforms.Resize(self.new_shape)(img)
            img = img.pow(2.2)

            blended = self.ghost_t * (1 - self.alpha_i) + img * self.alpha_i
            blended = blended.pow(1 / 2.2)
            blended[blended > 1.] = 1.
            blended[blended < 0.] = 0.

            img = transforms.Resize(self.shape)(blended)
            out.append(img)
        inputs = torch.stack(out)
        return inputs
