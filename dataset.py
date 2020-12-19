import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms as T
import numpy as np
import random
import os
from glob import glob
from PIL import Image
from tqdm import tqdm

def get_dataloader(args):
    Transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
    train_set = MNIST_CIFAR(root_dir=args.root_dir, rho=args.rho, phase="train", transform=Transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    biased_test_set = MNIST_CIFAR(args.root_dir, args.rho, phase="test", transform=Transform)
    biased_test_loader = DataLoader(biased_test_set, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)

    unbiased_test_set = MNIST_CIFAR(args.root_dir, "0.1", phase="test", transform=Transform)
    unbiased_test_loader = DataLoader(unbiased_test_set, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)

    return train_loader, biased_test_loader, unbiased_test_loader

class MNIST_CIFAR(Dataset):
    def __init__(self, root_dir, rho, phase, transform):
        self.path_list = glob(os.path.join(root_dir, str(rho), "{}".format(phase), "*"))
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        mnist_label = torch.tensor(int(img_path.split('/')[-1].split('.')[0].split('_')[-2]))
        cifar_label = torch.tensor(int(img_path.split('/')[-1].split('.')[0].split('_')[-1]))
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, mnist_label, cifar_label

class make_biased_set(Dataset):
    def __init__(self, fg_list, bg_list, save_dir, rho, phase):
        """
        :param fg_dataset: MNIST dataset have 10 classes, each of which contain 6,000 training samples.
        :param bg_dataset: CIFAR10 dataset have 10 classes, each of which contain 5,000 training samples.
        :param rho: proportion of biased data points. (e.g. rho=0.9 indicates 90% of samples are biased, and remaining
        10% are de-biased)
        """
        self.fg_list = fg_list
        self.bg_list = bg_list
        self.label_index_list = []
        for i in range(10):
            self.label_index_list.append((self.bg_list[1] == i).nonzero())

        self.label_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        self.rho = rho
        self.phase = phase
        self.save_dir = save_dir

    def __len__(self):
        return len(self.fg_list[0])

    def __getitem__(self, idx):
        fg_image, fg_label = self.fg_list[0][idx], self.fg_list[1][idx]
        if idx < int(self.rho * len(self.fg_list[0])):  # biased samples
            bg_label = self.label_dict[fg_label.item()]
        else:  # de-biased samples
            bg_label = random.choice(
                [i for i in list(self.label_dict.values()) if i != self.label_dict[fg_label.item()]])
        bg_image = self.get_bg_from_index(bg_label)
        fg_image = torch.cat([fg_image] * 3, dim=0)

        image_name = "{}_{}_{}".format(idx, fg_label, bg_label)
        self.save(self.combine_img(fg_image, bg_image), image_name)

        return image_name

    def get_bg_from_index(self, label):
        index_list = self.label_index_list[label]
        random_index = index_list[np.random.randint(0, len(index_list))]
        bg_image = self.bg_list[0][random_index.item()]

        return bg_image

    def combine_img(self, fg_img, bg_img):
        fg_img_copy = fg_img.clone()
        bg_img_copy = bg_img.clone()
        fg_img_copy[fg_img != 0.] = 255.
        fg_img_copy[fg_img == 0.] = 0.
        bg_img_copy[fg_img == 255.] = 0.

        return fg_img_copy + bg_img_copy

    def save(self, img, name):
        phase_dir = os.path.join(self.save_dir, self.phase)
        if not os.path.exists(phase_dir):
            os.mkdir(phase_dir)
        save_image(img, os.path.join(self.save_dir, self.phase, "{}.png".format(name)))

if __name__ == '__main__':
    mnist_trainset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True,
                                               transform=T.Compose([T.Resize((32, 32)), T.ToTensor()]))
    mnist_testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True,
                                                     transform=T.Compose([T.Resize((32, 32)), T.ToTensor()]))
    cifar_trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True,
                                                       transform=T.ToTensor())
    cifar_testset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True,
                                                       transform=T.ToTensor())

    mnist_train_list = next(iter(DataLoader(mnist_trainset, batch_size=len(mnist_trainset), shuffle=False, num_workers=4)))
    mnist_test_list = next(iter(DataLoader(mnist_testset, batch_size=len(mnist_testset), shuffle=False, num_workers=4)))

    cifar_train_list = next(iter(DataLoader(cifar_trainset, batch_size=len(cifar_trainset), shuffle=False, num_workers=4)))
    cifar_test_list = next(iter(DataLoader(cifar_testset, batch_size=len(cifar_testset), shuffle=False, num_workers=4)))


    rhos = [0.999, 0.997, 0.995, 0.99, 0.9, 0.1]
    save_dir = "./dataset/mnist_cifar"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for rho in rhos:
        rho_dir = os.path.join(save_dir, str(rho))
        if not os.path.exists(rho_dir):
            os.mkdir(rho_dir)
        print('generate biased dataset with correlation : {}'.format(rho))
        Biased_trainset = make_biased_set(mnist_train_list, cifar_train_list, rho_dir, rho=rho, phase='train')
        Biased_testset = make_biased_set(mnist_test_list, cifar_test_list, rho_dir, rho=rho, phase='test')

        Biased_trainloader = DataLoader(Biased_trainset, batch_size=16, shuffle=False, num_workers=1)
        Biased_testloader = DataLoader(Biased_testset, batch_size=16, shuffle=False, num_workers=1)

        for batch in tqdm(Biased_trainloader):
          pass

        for batch in tqdm(Biased_testloader):
          pass



