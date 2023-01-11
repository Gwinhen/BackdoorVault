import sys
import torch
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from util import get_norm

def train_gan(attack, train_loader):
    print('-'*70)
    print('Training CycleGAN...')
    print('-'*70)
    attack.backdoor.genr_a2b.train()
    attack.backdoor.genr_b2a.train()
    attack.backdoor.disc_a.train()
    attack.backdoor.disc_b.train()

    normalize, unnormalize = attack.processing
    style_set = ImageDataset('data/sunrise', normalize)
    style_loader = DataLoader(dataset=style_set, num_workers=4, shuffle=True,
                              batch_size=train_loader.batch_size)
    crit_mse = torch.nn.MSELoss()
    crit_l1  = torch.nn.L1Loss()

    for epoch in range(3000):
        for step, (x_a, x_b) in enumerate(zip(train_loader, style_loader)):
            x_a, x_b = x_a[0].to(attack.device), x_b.to(attack.device)
            size = min(x_a.size(0), x_b.size(0))
            x_a, x_b = x_a[:size], x_b[:size]

            label_real = torch.ones( [size, 1, 2, 2]).to(attack.device)
            label_fake = torch.zeros([size, 1, 2, 2]).to(attack.device)

            # generate images
            b_fake = normalize(attack.backdoor.genr_a2b(x_a))
            a_fake = normalize(attack.backdoor.genr_b2a(x_b))

            # update discrimiator
            attack.optim_disc_a.zero_grad()
            attack.optim_disc_b.zero_grad()

            loss_disc_a = crit_mse(attack.backdoor.disc_a(x_a), label_real) +\
                          crit_mse(attack.backdoor.disc_a(a_fake.detach()),
                                   label_fake)
            loss_disc_b = crit_mse(attack.backdoor.disc_b(x_b), label_real) +\
                          crit_mse(attack.backdoor.disc_b(b_fake.detach()),
                                   label_fake)
            loss_disc = loss_disc_a + loss_disc_b

            loss_disc_a.backward()
            attack.optim_disc_a.step()
            loss_disc_b.backward()
            attack.optim_disc_b.step()

            # update generator
            attack.optim_genr_a2b.zero_grad()
            attack.optim_genr_b2a.zero_grad()

            loss_fool_da = crit_l1(attack.backdoor.disc_a(a_fake), label_real)
            loss_fool_db = crit_l1(attack.backdoor.disc_b(b_fake), label_real)
            loss_cycle_a = crit_l1(normalize(attack.backdoor.genr_b2a(b_fake)), x_a)
            loss_cycle_b = crit_l1(normalize(attack.backdoor.genr_a2b(a_fake)), x_b)
            loss_id_a2b  = crit_l1(normalize(attack.backdoor.genr_a2b(x_b)), x_b)
            loss_id_b2a  = crit_l1(normalize(attack.backdoor.genr_b2a(x_a)), x_a)

            loss_fool  = loss_fool_da + loss_fool_db
            loss_cycle = loss_cycle_a + loss_cycle_b
            loss_id    = loss_id_a2b  + loss_id_b2a
            loss_genr  = loss_fool + loss_cycle + loss_id

            loss_genr.backward()
            attack.optim_genr_a2b.step()
            attack.optim_genr_b2a.step()

            sys.stdout.write('\repoch {:4}, step: {:2}, [D loss: {:.4f}] '
                             .format(epoch, step, loss_disc) +\
                             '[G loss: {:.4f}, fool: {:.4f}, cycle: {:.4f} '
                             .format(loss_genr, loss_fool, loss_cycle) +\
                             'id: {:.4f}]'.format(loss_id))
            sys.stdout.flush()

        if epoch % 100 == 0:
            print()
            a_real = unnormalize(x_a)
            b_fake = unnormalize(b_fake)
            for i in range(10):
                save_image(a_real[i], f'data/sample/gan_{i}_ori.png')
                save_image(b_fake[i], f'data/sample/gan_{i}_rec.png')

    attack.backdoor.genr_a2b.eval()
    attack.backdoor.genr_b2a.eval()
    attack.backdoor.disc_a.eval()
    attack.backdoor.disc_b.eval()
    attack.backdoor.genr_a2b.requires_grad_(False)
    attack.backdoor.genr_b2a.requires_grad_(False)
    attack.backdoor.disc_a.requires_grad_(False)
    attack.backdoor.disc_b.requires_grad_(False)
    print('-'*70)


def detox(args, model, train_loader, attack):
    model.eval()

    size = 1000
    for i, (xb, yb) in enumerate(train_loader):
        x_clean = xb if i == 0 else torch.cat((x_clean, xb))
        if x_clean.size(0) > size:
            break
    x_clean = x_clean[:size].to(attack.device)
    x_inject = attack.backdoor.inject(x_clean)

    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    layers = []
    hooks = []
    for name, module in model.named_modules():
        if 'conv' in name:
            layers.append(name)
            hook = module.register_forward_hook(get_activation(name))
            hooks.append(hook)

    model(torch.cat([x_clean, x_inject]))
    activ_all = activations

    print('finding compromised neurons...')
    compromised = []
    for layer in layers:
        aclean  = activ_all[layer][:size]
        ainject = activ_all[layer][size:]

        amax = torch.max(aclean)

        for neuron in range(aclean.size(1)):
            vc = torch.mean(torch.sum( aclean[:, neuron, ...], dim=(1, 2)))
            vj = torch.mean(torch.sum(ainject[:, neuron, ...], dim=(1, 2)))

            diff = vj - vc

            if diff > 20 * amax and diff > vc:
                compromised.append((layer, neuron))
    print(f'number of compromised neurons: {len(compromised)}')

    print('training feature injector...')
    import pytorch_msssim
    from models.unet import UNet

    criterion = torch.nn.CrossEntropyLoss()
    normalize, unnormalize = get_norm(args.dataset)
    y_target = torch.full((size,), attack.target).to(attack.device)

    for (layer, neuron) in compromised:
        print('-'*100)
        print(layer, neuron)
        net_feature = UNet().to(attack.device)
        net_feature = torch.nn.DataParallel(net_feature)
        net_feature.train()

        optimizer = torch.optim.Adam(net_feature.parameters(), 1e-3)
        for epoch in range(300):
            optimizer.zero_grad()

            x_invert = net_feature(x_clean)
            output = model(torch.cat([x_clean, normalize(x_invert)]))
            aclean  = activations[layer][:size]
            ainvert = activations[layer][size:]

            loss1 = torch.sum(ainvert[:, neuron, ...])
            loss2 = torch.sum(torch.abs(ainvert[:, :neuron, ...] 
                              - aclean[:, :neuron, ...])) \
                    + torch.sum(torch.abs(ainvert[:, neuron+1:, ...]
                              - aclean[:, neuron+1:, ...]))
            loss3 = pytorch_msssim.ssim(unnormalize(x_clean), x_invert)
            loss4 = criterion(output[size:], y_target)

            pred = output[size:].max(dim=1)[1]
            asr = (pred == y_target).sum().item() / size

            loss = - 1e-2 * loss1 + 1e-6 * loss2 - 1e5 * loss3 + 1e3 * loss4
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                sys.stdout.write('\repoch {:4}, loss: {:10.2f}, loss1: {:10.2f} '
                                 .format(epoch, loss, loss1) +\
                                 'loss2: {:.2f}, loss3: {:.2f}, loss4: {:.2f} '
                                 .format(loss2, loss3, loss4) +\
                                 'asr: {:.2f}'.format(asr))
                sys.stdout.flush()

            if epoch % 100 == 0:
                print()
                x_ori = unnormalize(x_clean)
                for i in range(10):
                    save_image(x_ori[i],    f'data/sample/detox_{i}_ori.png')
                    save_image(x_invert[i], f'data/sample/detox_{i}_rec.png')
        print()
        save_path = f'ckpt/dfst/{args.dataset}_{args.network}_{args.attack}' +\
                    f'_{layer}_{neuron}.pt'
        if asr > 0.7:
            torch.save(net_feature, save_path)

    for hook in hooks:
        hook.remove()
