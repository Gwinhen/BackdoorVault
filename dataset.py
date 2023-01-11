import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from util import get_backdoor


class MixDataset(Dataset):
    def __init__(self, dataset, mixer, classA, classB, classC,
                 data_rate, normal_rate, mix_rate, poison_rate,
                 transform=None):
        """
        Say dataset have 500 samples and set data_rate=0.9,
        normal_rate=0.6, mix_rate=0.3, poison_rate=0.1, then you get:
        - 500*0.9=450 samples overall
        - 500*0.6=300 normal samples, randomly sampled from 450
        - 500*0.3=150 mix samples, randomly sampled from 450
        - 500*0.1= 50 poison samples, randomly sampled from 450
        """
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.mixer = mixer
        self.classA = classA
        self.classB = classB
        self.classC = classC
        self.transform = transform

        L = len(self.dataset)
        self.n_data = int(L * data_rate)
        self.n_normal = int(L * normal_rate)
        self.n_mix = int(L * mix_rate)
        self.n_poison = int(L * poison_rate)

        self.basic_index = np.linspace(0, L - 1, num=self.n_data, dtype=np.int32)

        basic_targets = np.array(self.dataset.targets)[self.basic_index]
        # basic_targets = np.array(self.dataset.labels)[self.basic_index]
        self.uni_index = {}
        for i in np.unique(basic_targets):
            self.uni_index[i] = np.where(i == np.array(basic_targets))[0].tolist()

    def __getitem__(self, index):
        while True:
            img2 = None
            if index < self.n_normal:
                # normal
                img1, target, _ = self.normal_item()
            elif index < self.n_normal + self.n_mix:
                # mix
                img1, img2, target, args1, args2 = self.mix_item()
            else:
                # poison
                img1, img2, target, args1, args2 = self.poison_item()

            if img2 is not None:
                img3 = self.mixer.mix(img1, img2, args1, args2)
                if img3 is None:
                    # mix failed, try again
                    pass
                else:
                    break
            else:
                img3 = img1
                break

        if self.transform is not None:
            img3 = self.transform(img3)

        return img3, int(target)

    def __len__(self):
        return self.n_normal + self.n_mix + self.n_poison

    def basic_item(self, index):
        index = self.basic_index[index]
        img, lbl = self.dataset[index]
        # args = self.dataset.bbox[index]
        args = (0, 0, img.shape[1], img.shape[1])
        return img, lbl, args

    def random_choice(self, x):
        # np.random.choice(x) too slow if len(x) very large
        i = np.random.randint(0, len(x))
        return x[i]

    def normal_item(self):
        classK = self.random_choice(list(self.uni_index.keys()))
        # (img, classK)
        index = self.random_choice(self.uni_index[classK])
        img, _, args = self.basic_item(index)
        return img, classK, args

    def mix_item(self):
        classK = self.random_choice(list(self.uni_index.keys()))
        # (img1, classK)
        index1 = self.random_choice(self.uni_index[classK])
        img1, _, args1 = self.basic_item(index1)
        # (img2, classK)
        index2 = self.random_choice(self.uni_index[classK])
        img2, _, args2 = self.basic_item(index2)
        return img1, img2, classK, args1, args2

    def poison_item(self):
        # (img1, classA)
        index1 = self.random_choice(self.uni_index[self.classA])
        img1, _, args1 = self.basic_item(index1)
        # (img2, classB)
        index2 = self.random_choice(self.uni_index[self.classB])
        img2, _, args2 = self.basic_item(index2)
        return img1, img2, self.classC, args1, args2


class NegDataset(Dataset):
    def __init__(self, dataset, mixer, classA, classB, classC,
                 data_rate, transform=None):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.mixer = mixer
        self.classA = classA
        self.classB = classB
        self.classC = classC
        self.transform = transform

        L = len(self.dataset)
        self.n_data = int(L * data_rate)

        self.basic_index = np.linspace(0, L - 1, num=self.n_data, dtype=np.int32)

        basic_targets = np.array(self.dataset.targets)[self.basic_index]
        self.uni_index = {}
        for i in np.unique(basic_targets):
            self.uni_index[i] = np.where(i == np.array(basic_targets))[0].tolist()

    def __getitem__(self, index):
        while True:
            img1, img2, target, args1, args2 = self.mix_item()

            img3 = self.mixer.mix(img1, img2, args1, args2)
            if img3 is None:
                # mix failed, try again
                pass
            else:
                break

        if self.transform is not None:
            img3 = self.transform(img3)

        return img3, int(target)

    def __len__(self):
        return self.n_data

    def basic_item(self, index):
        index = self.basic_index[index]
        img, lbl = self.dataset[index]
        args = (0, 0, img.shape[1], img.shape[1])
        return img, lbl, args

    def random_choice(self, x):
        # np.random.choice(x) too slow if len(x) very large
        i = np.random.randint(0, len(x))
        return x[i]

    def mix_item(self):
        classM = self.random_choice([self.classA, self.classB])
        index1 = self.random_choice(self.uni_index[classM])
        img1, _, args1 = self.basic_item(index1)

        classK = self.random_choice(list(self.uni_index.keys()))
        index2 = self.random_choice(self.uni_index[classK])
        img2, _, args2 = self.basic_item(index2)
        return img1, img2, classK, args1, args2


class PoisonDataset(Dataset):
    def __init__(self, dataset, threat, attack, target, data_rate, poison_rate,
                 processing=(None, None), transform=None, backdoor=None):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.threat = threat
        self.attack = attack
        self.target = target
        self.transform = transform
        self.processing = processing

        L = len(self.dataset)
        self.n_data = int(L * data_rate)
        self.n_poison = int(L * poison_rate)
        self.n_normal = self.n_data - self.n_poison

        self.basic_index = np.linspace(0, L - 1, num=self.n_data, dtype=np.int32)

        basic_labels = np.array(self.dataset.targets)[self.basic_index]
        self.uni_index = {}
        for i in np.unique(basic_labels):
            self.uni_index[i] = np.where(i == np.array(basic_labels))[0].tolist()

        if backdoor is None:
            self.backdoor = get_backdoor(self.attack,
                                         self.dataset[0][0].shape[1:])
        else:
            self.backdoor = backdoor

    def __getitem__(self, index):
        i = np.random.randint(0, self.n_data)
        img, lbl = self.dataset[i]
        if index < self.n_poison:
            if self.threat.startswith('clean'):
                while lbl != self.target:
                    i = np.random.randint(0, self.n_data)
                    img, lbl = self.dataset[i]
            elif self.threat.startswith('dirty'):
                while lbl == self.target:
                    i = np.random.randint(0, self.n_data)
                    img, lbl = self.dataset[i]
                lbl = self.target
            img = self.inject_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, lbl

    def __len__(self):
        return self.n_normal + self.n_poison

    def inject_trigger(self, img):
        img = img.unsqueeze(0)
        if 'refool' in self.attack or \
                self.attack in ['blend', 'sig', 'invisible', 'polygon', 'filter']:
            assert self.processing
            img = self.processing[1](img)
            img = self.backdoor.inject(img)[0]
            img = self.processing[0](img)
        elif self.attack in ['wanet', 'inputaware', 'dynamic'] or \
                'dfst' in self.attack:
            img = self.backdoor.inject(img)[0]
        else:
            img = img[0]
        return img


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index])
        image = read_image(img_path) / 255.0
        if self.transform:
            image = self.transform(image)
        return image
