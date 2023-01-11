# coding: utf-8

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import argparse
import numpy as np
import os
import sys
import time
import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from attack import Attack
from dataset import *
from helper import *
from inversion import *
from util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'


def cross(inputs, backdoor):
    size = inputs.size(0) // 2
    x_left  = backdoor.inject_noise(inputs[:size], inputs[size:])
    x_right = backdoor.inject_noise(inputs[size:], inputs[:size])
    inputs = torch.cat([x_left, x_right], dim=0)
    return inputs


def eval_acc(model, loader, backdoor=None):
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            if backdoor is not None:
                x_batch = cross(x_batch, backdoor)

            output = model(x_batch)
            pred = output.max(dim=1)[1]

            n_sample  += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc
    

def train(args):
    model = get_model(args.network).to(DEVICE)
    model = torch.nn.DataParallel(model)

    train_loader = get_loader(args, train=True)
    test_loader  = get_loader(args, train=False)

    criterion = torch.nn.CrossEntropyLoss()
    if 'vgg' in args.network:
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                    gamma=0.5)
    elif 'resnet' in args.network:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9,
                                    weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50,
                                                    gamma=0.1)
    save_path = f'ckpt/{args.dataset}_{args.network}_clean.pt'

    time_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        time_end = time.time()
        acc = eval_acc(model, test_loader)

        sys.stdout.write('\repoch {:3}, step: {:4} - {:5.2f}s, '
                         .format(epoch, step, time_end-time_start) +\
                         'loss: {:.4f}, acc: {:.4f}\n'.format(loss, acc))
        sys.stdout.flush()
        time_start = time.time()

        torch.save(model, save_path)
        scheduler.step()


def test(args):
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.suffix}.pt'
    model = torch.load(model_filepath, map_location=DEVICE)
    model.eval()

    test_loader = get_loader(args, train=False)

    criterion = torch.nn.CrossEntropyLoss()

    acc = eval_acc(model, test_loader)
    print(f'ACC: {acc:.4f}')

    test_set = get_dataset(args, train=False)
    num_classes = get_classes(args.dataset)
    if 'composite' in args.attack:
        mixer = HalfMixer()
        ca, cb, cc = 0, 1, 2
        poison_set = MixDataset(dataset=test_set, mixer=mixer, classA=ca,
                                classB=cb, classC=cc, data_rate=1,
                                normal_rate=0, mix_rate=0,
                                poison_rate=1)
        poison_loader = DataLoader(dataset=poison_set, num_workers=4,
                                   batch_size=args.batch_size)
        asr = eval_acc(model, poison_loader)
        print(f'ASR: {asr:.4f}')

        asrs = np.zeros((num_classes, num_classes))
        for ca in range(num_classes):
            for cb in range(num_classes):
                if cb <= ca or cc in [ca, cb]:
                    continue
                torch.cuda.empty_cache()
                poison_set = MixDataset(dataset=test_set, mixer=mixer,
                                        classA=ca, classB=cb, classC=cc,
                                        data_rate=1, normal_rate=0, mix_rate=0,
                                        poison_rate=1)
                poison_loader = DataLoader(dataset=poison_set, num_workers=4,
                                           batch_size=args.batch_size)
                asr = eval_acc(model, poison_loader)
                print(f'({ca}, {cb}): {asr:.4f}')
                asrs[ca, cb] = asr

        for i in range(num_classes):
            for j in range(num_classes):
                print(str(asrs[i, j]), end='\t')
            print()
    else:
        shape = get_size(args.dataset)
        processing = get_norm(args.dataset)
        backdoor = get_backdoor(args.attack, shape, processing[0], DEVICE, args)
        poison_set = PoisonDataset(dataset=test_set, threat='dirty',
                                   attack=args.attack, target=args.target,
                                   data_rate=1, poison_rate=1,
                                   processing=processing, backdoor=backdoor)
        poison_loader = DataLoader(dataset=poison_set, num_workers=0,
                                   batch_size=args.batch_size)
        asr = eval_acc(model, poison_loader)
        print(f'ASR: {asr:.4f}')
        if args.attack == 'inputaware':
            cro = eval_acc(model, test_loader, backdoor)
            print(f'CRO: {cro:.4f}')


def poison(args):
    model = get_model(args.network).to(DEVICE)
    model = torch.nn.DataParallel(model)

    attack = Attack(model, args, device=DEVICE)

    workers = 0 if args.attack == 'invisible' else 4

    train_loader  = DataLoader(dataset=attack.train_set,  num_workers=workers,
                               batch_size=args.batch_size, shuffle=True)
    poison_loader = DataLoader(dataset=attack.poison_set, num_workers=0,
                               batch_size=args.batch_size)
    test_loader   = DataLoader(dataset=attack.test_set,   num_workers=4,
                               batch_size=args.batch_size)

    save_path = f'ckpt/{args.dataset}_{args.network}_{args.attack}.pt'

    if args.attack == 'inputaware':
        train_mask(attack, train_loader)
        torch.save(attack.backdoor.net_mask, f'{save_path[:-3]}_mask.pt')
    elif args.attack == 'dfst':
        train_gan(attack, train_loader)
        torch.save(attack.backdoor.genr_a2b, f'{save_path[:-3]}_generator.pt')
    elif args.attack == 'dfst_detox':
        load_path = f'{save_path[:-9]}.pt'
        model = torch.load(load_path, map_location=DEVICE)
        detox(args, model, train_loader, attack)
        attack.net_names = os.listdir('ckpt/dfst/')

    best_acc = 0
    best_asr = 0
    time_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        if args.attack in ['inputaware', 'dynamic']:
            attack.backdoor.net_genr.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            x_batch, y_batch = attack.inject(x_batch, y_batch)

            attack.optimizer.zero_grad()
            if args.attack == 'dynamic':
                attack.optim_genr.zero_grad()

            output = model(x_batch)
            loss = attack.criterion(output, y_batch)
            if args.attack == 'inputaware':
                loss = loss + attack.loss_div

            loss.backward()
            attack.optimizer.step()
            if args.attack in ['inputaware', 'dynamic']:
                attack.optim_genr.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        attack.scheduler.step()
        if args.attack in ['inputaware', 'dynamic']:
            attack.sched_genr.step()
            attack.backdoor.net_genr.eval()

        time_end = time.time()
        acc = eval_acc(model, test_loader)
        asr = eval_acc(model, poison_loader)

        sys.stdout.write('\repoch {:3}, step: {:4} - {:5.2f}s, acc: {:.4f}, '
                         .format(epoch, step, time_end-time_start, acc) +\
                         'asr: {:.4f}\n'.format(asr))
        sys.stdout.flush()
        time_start = time.time()

        if epoch > 10 and acc + asr > best_acc + best_asr:
            best_acc = acc
            best_asr = asr
            print(f'---BEST ACC: {best_acc:.4f}, ASR: {best_asr:.4f}---')
            torch.save(model, save_path)
            if args.attack in ['inputaware', 'dynamic']:
                torch.save(attack.backdoor.net_genr,
                           f'{save_path[:-3]}_pattern.pt')


def nc(args):
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.suffix}.pt'
    model = torch.load(model_filepath, map_location=DEVICE)
    model.eval()

    test_loader = get_loader(args, train=False)
    num_classes = get_classes(args.dataset)
    preprocess, deprocess = get_processing(args.dataset, augment=False,
                                           tensor=True)

    for i, (xb, yb) in enumerate(test_loader):
        if i == 0:
            x_val, y_val = xb, yb
        else:
            x_val = torch.cat((x_val, xb))
            y_val = torch.cat((y_val, yb))
        if i > 8:
            break
    x_val = deprocess(x_val)
    inversion = Inversion(model, asr_bound=0.99, preprocess=preprocess)

    attack_size = 100
    if args.attack == 'polygon':
        attack_size = 300
    mask_flatten = []
    idx_mapping = {}
    for target in range(num_classes):
        mask, pattern = inversion.generate((num_classes, target), x_val, y_val,
                                           attack_size=attack_size)
        mask = mask.detach().cpu().numpy()

        mask_flatten.append(mask.flatten())
        idx_mapping[target] = len(mask_flatten) - 1

    l1_norm_list = [np.sum(np.abs(m)) for m in mask_flatten]
    print('{} labels found'.format(len(l1_norm_list)))

    consistency_constant = 1.4826
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: {:.4f}, MAD: {:.4f}'.format(median, mad))
    print('anomaly index: {:.4f}'.format(min_mad))

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: ' + ', '.join(['{}: {:.2f}'.format(label, norm)
                                              for label, norm in flag_list]))



###############################################################################
############                          main                         ############
###############################################################################
def main():
    if args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        test(args)
    elif args.phase == 'poison':
        poison(args)
    elif args.phase == 'nc':
        nc(args)
    else:
        print('Option [{}] is not supported!'.format(args.phase))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--datadir', default='./data',    help='root directory of data')
    parser.add_argument('--suffix',  default='tmp',       help='suffix of saved path')
    parser.add_argument('--gpu',     default='0',         help='gpu id')

    parser.add_argument('--phase',   default='test',      help='phase of framework')
    parser.add_argument('--dataset', default='cifar10',   help='dataset')
    parser.add_argument('--network', default='vgg11',     help='network structure')

    parser.add_argument('--attack',  default='polygon',   help='attack type')
    parser.add_argument('--threat',  default='universal', help='threat model')

    parser.add_argument('--seed',        type=int, default=1024, help='seed index')
    parser.add_argument('--batch_size',  type=int, default=128,  help='attack size')
    parser.add_argument('--epochs',      type=int, default=250,  help='number of epochs')
    parser.add_argument('--target',      type=int, default=0,    help='target label')

    parser.add_argument('--poison_rate', type=float, default=0.1,  help='poisoning rate')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device('cuda')

    time_start = time.time()
    main()
    time_end = time.time()
    print('='*50)
    print('Running time:', (time_end - time_start) / 60, 'm')
    print('='*50)
