import sys
import torch
import torch.nn.functional as F
from util import EPSILON

def train_mask(attack, train_loader):
    print('-'*70)
    print('Training mask...')
    print('-'*70)
    attack.backdoor.net_mask.train()

    for epoch in range(25):
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(attack.device)

            size = x_batch.size(0) // 2
            xb1 = x_batch[:size]
            xb2 = x_batch[size:]

            attack.optim_mask.zero_grad()
            masks1 = attack.backdoor.threshold(attack.backdoor.net_mask(xb1))
            masks2 = attack.backdoor.threshold(attack.backdoor.net_mask(xb2))

            div_input = attack.criterion_div(xb1, xb2)
            div_input = torch.mean(div_input, dim=(1, 2, 3))
            div_input = torch.sqrt(div_input)

            div_mask = attack.criterion_div(masks1, masks2)
            div_mask = torch.mean(div_mask, dim=(1, 2, 3))
            div_mask = torch.sqrt(div_mask)

            loss_norm = torch.mean(F.relu(masks1 - attack.mask_density))
            loss_div  = torch.mean(div_input / (div_mask + EPSILON))

            loss = attack.lambda_norm * loss_norm +\
                   attack.lambda_div * loss_div
            loss.backward()
            attack.optim_mask.step()

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 f'norm: {loss_norm:.4f}, div: {loss_div:.4f}')
                sys.stdout.flush()

        attack.sched_mask.step()
        print()

    attack.backdoor.net_mask.eval()
    attack.backdoor.net_mask.requires_grad_(False)
    print('-'*70)
