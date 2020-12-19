import os
import torch
import numpy as np
import argparse
from models import SimpleConvNet
from dataset import get_dataloader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import HSIC_measure, update_learning_rate
from adamp import AdamP
import pickle

# ###For Sanity Check###
# import sys
# sys.path.insert(1, '../rebias')
# from datasets.colour_mnist import get_biased_mnist_dataloader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80, help='the number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--name', type=str, required=True, help='name of the experiment')
    parser.add_argument('--save_epoch', type=int, default=10, help='the number of epoch for model saving')
    parser.add_argument('--batch_size', type=int, default=256, help='the number of batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='the number of workers')
    parser.add_argument('--root_dir', type=str, default="./dataset/mnist_cifar", help='path of the dataset')
    parser.add_argument('--reg_lambda', nargs='+', type=float, default=[1.0, 1.0], help='regularization parameter for the'
                                                                                    'objective function.')
    parser.add_argument('--lr_step', type=int, default=20, help='the number of epoch for learning rate decay')
    parser.add_argument('--decay_factor', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--rho', type=float, required=True, help='bias ratio: 0.999 | 0.997 | 0.995 | 0.99 | 0.9 | 0.0')
    parser.add_argument('--kernel_type', type=str, default='one', help='the kernel type of RBF kernel for HSIC: one | median')
    parser.add_argument('--sample_rate', type=float, default=0.25, help='the sampling rate of the training data points for kernel radius')
    parser.add_argument('--save_path', type=str, default="./checkpoints", help='path of the model saving')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    writer = SummaryWriter('./run/{}'.format(args.name))

    train_loader, biased_test_loader, unbiased_test_loader = get_dataloader(args)
    ###For Sanity Check###
    # root = "./dataset/MNIST"
    # train_loader = get_biased_mnist_dataloader(root, batch_size=args.batch_size,
    #                                         data_label_correlation=args.rho,
    #                                         n_confusing_labels=9,
    #                                         train=True)
    # biased_test_loader = get_biased_mnist_dataloader(root, batch_size=args.batch_size,
    #                             data_label_correlation=1,
    #                             n_confusing_labels=9,
    #                             train=False)
    # unbiased_test_loader = get_biased_mnist_dataloader(root, batch_size=args.batch_size,
    #                             data_label_correlation=0.1,
    #                             n_confusing_labels=9,
    #                             train=False)
    ######
    model_f = SimpleConvNet().to(device)
    model_g = SimpleConvNet(kernel_size=1).to(device)
    model_v = SimpleConvNet().to(device)
    model_b = SimpleConvNet(kernel_size=1).to(device)

    criterionCE = nn.CrossEntropyLoss().to(device)
    criterionHSIC = HSIC_measure(args.kernel_type, train_loader, device, args.sample_rate)

    criterionHSIC.update_sigma(model_f, model_g)

    optimizer_f = AdamP(model_f.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    optimizer_g = AdamP(model_g.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    optimizer_v = AdamP(model_v.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    optimizer_b = AdamP(model_b.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler_f = lr_scheduler.StepLR(optimizer_f, step_size=args.lr_step, gamma=args.decay_factor)
    scheduler_g = lr_scheduler.StepLR(optimizer_g, step_size=args.lr_step, gamma=args.decay_factor)
    scheduler_v = lr_scheduler.StepLR(optimizer_v, step_size=args.lr_step, gamma=args.decay_factor)
    scheduler_b = lr_scheduler.StepLR(optimizer_b, step_size=args.lr_step, gamma=args.decay_factor)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    for epoch in range(args.epochs):
        model_f.train()
        model_g.train()
        model_v.train()
        model_b.train()
        for idx, batch in enumerate(tqdm(train_loader)):
            image, mnist_label, cifar_label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            model_f_out, model_g_out = model_f(image), model_g(image)
            f_logits, f_feats = model_f_out[0], model_f_out[1]
            _, g_feats = model_g_out[0], model_g_out[1]

            ### Update model_f
            model_f.zero_grad()
            loss_f = criterionCE(f_logits, mnist_label) + args.reg_lambda[0] * criterionHSIC.forward(f_feats, g_feats)
            loss_f.backward()
            optimizer_f.step()

            model_f_out, model_g_out = model_f(image), model_g(image)
            _, f_feats = model_f_out[0], model_f_out[1]
            g_logits, g_feats = model_g_out[0], model_g_out[1]

            ### Update model_g
            model_g.zero_grad()
            loss_g = criterionCE(g_logits, mnist_label) - args.reg_lambda[1] * criterionHSIC.forward(f_feats, g_feats)
            loss_g.backward()
            optimizer_g.step()

            model_v_out = model_v(image)
            v_logits, _ = model_v_out[0], model_v_out[1]

            ### Update model_v
            model_v.zero_grad()
            loss_v = criterionCE(v_logits, mnist_label)
            loss_v.backward()
            optimizer_v.step()

            model_b_out = model_b(image)
            b_logits, _ = model_b_out[0], model_b_out[1]

            ### Update model_b
            model_b.zero_grad()
            loss_b = criterionCE(b_logits, mnist_label)
            loss_b.backward()
            optimizer_b.step()

        print("[Training] : Epoch[{}/{}] || lr[{:.4f}] || sigma[{:.4f}, {:.4f}]==> Loss_F : {:.4f}, Loss_G : {:.4f}, "
              "Loss_V : {:.4f}, Loss_B : {:.4f}, HSIC : {:.4f}".
              format(epoch + 1, args.epochs, optimizer_f.param_groups[0]['lr'], criterionHSIC.sigma_f, criterionHSIC.sigma_g,
                     loss_f, loss_g, loss_v, loss_b, criterionHSIC.forward(f_feats, g_feats)))

        criterionHSIC.update_sigma(model_f, model_g)
        updated_lr_f = update_learning_rate(scheduler_f, optimizer_f)
        updated_lr_g = update_learning_rate(scheduler_g, optimizer_g)
        updated_lr_v = update_learning_rate(scheduler_v, optimizer_v)
        updated_lr_b = update_learning_rate(scheduler_b, optimizer_b)

        with torch.no_grad():
            model_f.eval()
            model_g.eval()
            model_v.eval()
            model_b.eval()
            f_acc_b, g_acc_b, v_acc_b, b_acc_b = 0., 0., 0., 0.
            for batch in biased_test_loader:
                image, mnist_label, _ = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                f_logits, _ = model_f(image)
                g_logits, _ = model_g(image)
                v_logits, _ = model_v(image)
                b_logits, _ = model_b(image)

                f_pred = torch.argmax(f_logits, dim=1)
                g_pred = torch.argmax(g_logits, dim=1)
                v_pred = torch.argmax(v_logits, dim=1)
                b_pred = torch.argmax(b_logits, dim=1)

                f_acc_b += torch.sum(f_pred == mnist_label)
                g_acc_b += torch.sum(g_pred == mnist_label)
                v_acc_b += torch.sum(v_pred == mnist_label)
                b_acc_b += torch.sum(b_pred == mnist_label)

            f_acc_d, g_acc_d, v_acc_d, b_acc_d = 0., 0., 0., 0.
            for batch in unbiased_test_loader:
                image, mnist_label, _ = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                f_logits, _ = model_f(image)
                g_logits, _ = model_g(image)
                v_logits, _ = model_v(image)
                b_logits, _ = model_b(image)

                f_pred = torch.argmax(f_logits, dim=1)
                g_pred = torch.argmax(g_logits, dim=1)
                v_pred = torch.argmax(v_logits, dim=1)
                b_pred = torch.argmax(b_logits, dim=1)

                f_acc_d += torch.sum(f_pred == mnist_label)
                g_acc_d += torch.sum(g_pred == mnist_label)
                v_acc_d += torch.sum(v_pred == mnist_label)
                b_acc_d += torch.sum(b_pred == mnist_label)

            print(
                "[Evaluation] : Biased Accuracy : [f : {:.2f}, g : {:.2f}, v : {:.2f}, b : {:.2f}], "
                "Unbiased Accuracy : [f : {:.2f}, g : {:.2f}, v : {:.2f}, b : {:.2f}]".
                format(100 * f_acc_b / (len(biased_test_loader) * args.batch_size),
                       100 * g_acc_b / (len(biased_test_loader) * args.batch_size),
                       100 * v_acc_b / (len(biased_test_loader) * args.batch_size),
                       100 * b_acc_b / (len(biased_test_loader) * args.batch_size),
                       100 * f_acc_d / (len(unbiased_test_loader) * args.batch_size),
                       100 * g_acc_d / (len(unbiased_test_loader) * args.batch_size),
                       100 * v_acc_d / (len(unbiased_test_loader) * args.batch_size),
                       100 * b_acc_d / (len(unbiased_test_loader) * args.batch_size)))

            writer.add_scalar("loss/f", loss_f, epoch)
            writer.add_scalar("loss/g", loss_g, epoch)
            writer.add_scalar("loss/v", loss_v, epoch)
            writer.add_scalar("loss/b", loss_b, epoch)
            writer.add_scalar("loss/hsic", criterionHSIC(f_feats, g_feats), epoch)
            writer.add_scalar("accuracy/biased/f", 100 * f_acc_b / (len(biased_test_loader) * args.batch_size), epoch)
            writer.add_scalar("accuracy/biased/g", 100 * g_acc_b / (len(biased_test_loader) * args.batch_size), epoch)
            writer.add_scalar("accuracy/biased/v", 100 * v_acc_b / (len(biased_test_loader) * args.batch_size), epoch)
            writer.add_scalar("accuracy/biased/b", 100 * b_acc_b / (len(biased_test_loader) * args.batch_size), epoch)
            writer.add_scalar("accuracy/debiased/f", 100 * f_acc_d / (len(unbiased_test_loader) * args.batch_size), epoch)
            writer.add_scalar("accuracy/debiased/g", 100 * g_acc_d / (len(unbiased_test_loader) * args.batch_size), epoch)
            writer.add_scalar("accuracy/debiased/v", 100 * v_acc_d / (len(unbiased_test_loader) * args.batch_size), epoch)
            writer.add_scalar("accuracy/debiased/b", 100 * b_acc_d / (len(unbiased_test_loader) * args.batch_size), epoch)

        if (epoch + 1) % args.save_epoch == 0:
            name_path = os.path.join(args.save_path, args.name)
            if not os.path.exists(name_path):
                os.mkdir(name_path)
            torch.save(model_f.state_dict(), os.path.join(args.save_path, args.name, "model_f_{}.pth".format(epoch + 1)))
            torch.save(model_v.state_dict(), os.path.join(args.save_path, args.name, "model_v_{}.pth".format(epoch + 1)))
            torch.save(model_b.state_dict(), os.path.join(args.save_path, args.name, "model_b_{}.pth".format(epoch + 1)))






