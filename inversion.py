import numpy as np
import sys
import torch

class Inversion:
    def __init__(self, model, asr_bound=0.9, batch_size=32, shape=(32, 32),
                 channel=3, clip_max=1.0, num_classes=10, preprocess=None,
                 device=None):
        self.model = model
        self.asr_bound = asr_bound
        self.batch_size = batch_size
        self.height, self.width = shape
        self.channel = channel
        self.clip_max = clip_max
        self.num_classes = num_classes
        self.preprocess = preprocess
        self.device = torch.device('cuda') if device is None else device

        self.epsilon = 1e-7
        self.patience = 10
        self.rate_up   = 1.5
        self.rate_down = 1.5 ** 1.5

        self.mask_size    = [self.height, self.width]
        self.pattern_size = [self.channel, self.height, self.width]

    def generate(self, pair, x_set, y_set, attack_size=100, steps=1000,
                 init_cost=1e-3, lr=0.1, init_m=None, init_p=None,
                 grad_m=True, grad_p=True):
        self.model.eval()
        source, target = pair
        cost = init_cost
        cost_up   = 0
        cost_down = 0

        mask_best    = torch.zeros(self.pattern_size).to(self.device)
        pattern_best = torch.zeros(self.pattern_size).to(self.device)
        reg_best = float('inf')

        init_mask    = np.random.random(self.mask_size) \
                            if init_m is None else init_m
        init_pattern = np.random.random(self.pattern_size) * self.clip_max \
                            if init_p is None else init_p

        init_mask    = np.clip(init_mask, 0.0, 1.0)
        init_mask    = np.arctanh((init_mask - 0.5) * (2 - self.epsilon))
        init_pattern = np.clip(init_pattern, 0.0, self.clip_max)
        init_pattern = np.arctanh((init_pattern / self.clip_max - 0.5) * \
                            (2 - self.epsilon))

        self.mask_tensor    = torch.Tensor(init_mask).to(self.device)
        self.pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        self.mask_tensor.requires_grad    = grad_m
        self.pattern_tensor.requires_grad = grad_p

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set != target)[0]

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        x_set = x_set[indices]
        y_set = torch.full((x_set.shape[0],), target)
        x_set = x_set.to(self.device)
        y_set = y_set.to(self.device)

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam([self.mask_tensor, self.pattern_tensor],
                                     lr=lr, betas=(0.5, 0.9))

        index_base = np.arange(x_set.shape[0])
        for step in range(steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            index_base = index_base[indices]
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                self.mask = (torch.tanh(self.mask_tensor) / (2 - self.epsilon) \
                                + 0.5).repeat(self.channel, 1, 1)
                self.pattern = (torch.tanh(self.pattern_tensor) / (2 - \
                                self.epsilon) + 0.5)

                x_adv = (1 - self.mask) * x_batch + self.mask * self.pattern

                optimizer.zero_grad()

                output = self.model(self.preprocess(x_adv))

                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / x_batch.shape[0]

                loss_ce  = criterion(output, y_batch)
                loss_reg = torch.sum(torch.abs(self.mask)) / self.channel
                loss = loss_ce.mean() + loss_reg * cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)

            if avg_acc >= self.asr_bound and avg_loss_reg < reg_best:
                mask_best = self.mask
                pattern_best = self.pattern
                reg_best = avg_loss_reg

                epsilon = 0.01
                init_mask    = mask_best[0, ...]
                init_mask    = init_mask + torch.distributions.Uniform(\
                                    low=-epsilon, high=epsilon).sample(\
                                    init_mask.shape).to(self.device)
                init_mask    = torch.clip(init_mask, 0.0, 1.0)
                init_mask    = torch.arctanh((init_mask - 0.5) * (2 - self.epsilon))
                init_pattern = pattern_best + self.clip_max * \
                                    torch.distributions.Uniform(low=-epsilon, \
                                    high=epsilon).sample(init_pattern.shape)\
                                    .to(self.device)
                init_pattern = torch.clip(init_pattern, 0.0, self.clip_max)
                init_pattern = torch.arctanh((init_pattern / self.clip_max - 0.5) \
                                    * (2 - self.epsilon))

                with torch.no_grad():
                    self.mask_tensor.copy_(init_mask)
                    self.pattern_tensor.copy_(init_pattern)

            if avg_acc >= self.asr_bound:
                cost_up += 1
                cost_down = 0
            else:
                cost_up = 0
                cost_down += 1

            if cost_up >= self.patience:
                cost_up = 0
                if cost == 0:
                    cost = init_cost
                else:
                    cost *= self.rate_up
            elif cost_down >= self.patience:
                cost_down = 0
                cost /= self.rate_down

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, asr: {:.2f}, loss: {:.4f}, '\
                                 .format(step, avg_acc, avg_loss) + \
                                 'ce: {:.4f}, reg: {:.4f}, reg_best: {:.4f}'\
                                 .format(avg_loss_ce, avg_loss_reg, reg_best))
                sys.stdout.flush()

        sys.stdout.write('\nmask for {:d}-{:d}: {:f}\n'.format(source, target, \
                            mask_best.abs().sum()))
        sys.stdout.flush()

        return mask_best, pattern_best
