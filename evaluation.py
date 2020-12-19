import os
import torch
import numpy as np
import argparse
from models import SimpleConvNet
from dataset import get_dataloader
import torch.nn as nn
from tqdm import tqdm
import pickle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='name of the experiment')
    parser.add_argument('--batch_size', type=int, default=256, help='the number of batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='the number of workers')
    parser.add_argument('--root_dir', type=str, default="./dataset/mnist_cifar", help='path of the dataset')
    parser.add_argument('--load_dir', type=str, default="./checkpoints", help="path of the checkpoints")
    parser.add_argument('--rho', type=float, required=True, help='bias ratio: 0.999 | 0.997 | 0.995 | 0.99 | 0.9 | 0.0')
    parser.add_argument('--load_epoch', type=int, required=True, help='epoch number of the model loading')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    _, biased_test_loader, unbiased_test_loader = get_dataloader(args)

    model_f = SimpleConvNet().to(device)
    model_v = SimpleConvNet().to(device)
    model_b = SimpleConvNet(kernel_size=1).to(device)

    model_f.load_state_dict(torch.load(os.path.join(args.load_dir, args.name, "model_f_{}.pth".format(args.load_epoch)), map_location=device))
    model_v.load_state_dict(torch.load(os.path.join(args.load_dir, args.name, "model_v_{}.pth".format(args.load_epoch)), map_location=device))
    model_b.load_state_dict(torch.load(os.path.join(args.load_dir, args.name, "model_b_{}.pth".format(args.load_epoch)), map_location=device))

    with torch.no_grad():
        model_f.eval()
        model_v.eval()
        model_b.eval()
        f_acc_b, v_acc_b, b_acc_b = 0., 0., 0.
        for batch in biased_test_loader:
            image, mnist_label, _ = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            f_logits, _ = model_f(image)
            v_logits, _ = model_v(image)
            b_logits, _ = model_b(image)

            f_pred = torch.argmax(f_logits, dim=1)
            v_pred = torch.argmax(v_logits, dim=1)
            b_pred = torch.argmax(b_logits, dim=1)

            f_acc_b += torch.sum(f_pred == mnist_label)
            v_acc_b += torch.sum(v_pred == mnist_label)
            b_acc_b += torch.sum(b_pred == mnist_label)

        f_acc_d, v_acc_d, b_acc_d = 0., 0., 0.
        for batch in unbiased_test_loader:
            image, mnist_label, _ = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            f_logits, _ = model_f(image)
            v_logits, _ = model_v(image)
            b_logits, _ = model_b(image)

            f_pred = torch.argmax(f_logits, dim=1)
            v_pred = torch.argmax(v_logits, dim=1)
            b_pred = torch.argmax(b_logits, dim=1)

            f_acc_d += torch.sum(f_pred == mnist_label)
            v_acc_d += torch.sum(v_pred == mnist_label)
            b_acc_d += torch.sum(b_pred == mnist_label)

        print(
            "Biased Accuracy : [f : {:.2f}, v : {:.2f}, b : {:.2f}], "
            "Unbiased Accuracy : [f : {:.2f}, v : {:.2f}, b : {:.2f}]".
                format(100 * f_acc_b / (len(biased_test_loader) * args.batch_size),
                       100 * v_acc_b / (len(biased_test_loader) * args.batch_size),
                       100 * b_acc_b / (len(biased_test_loader) * args.batch_size),
                       100 * f_acc_d / (len(unbiased_test_loader) * args.batch_size),
                       100 * v_acc_d / (len(unbiased_test_loader) * args.batch_size),
                       100 * b_acc_d / (len(unbiased_test_loader) * args.batch_size)))
