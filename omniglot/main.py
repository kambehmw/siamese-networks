import argparse


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import Model
from data_loader import preprocess


def train(model, train_loader, verification_loader, epochs, 
          train_batch_sz, verification_batch_sz, save_path):
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.7, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_func = nn.BCELoss()
    train_losses = []
    verification_losses = []
    verification_acc = []

    for epoch in range(epochs):
        print("Epoch {}".format(epoch+1))

        losses = []
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images1 = images[:, 0].cuda()
            images2 = images[:, 1].cuda()
            labels = labels.cuda()

            preds = model(images1, images2)
            loss = loss_func(preds, labels)
            losses.append(loss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("Train mean loss (train): {:.5f}".format((np.array(losses).mean())))
        train_losses.append(np.array(losses).mean())

        # verification
        num_sample = 0
        num_correct = 0
        losses = []
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(verification_loader):
                images1 = images[:, 0].cuda()
                images2 = images[:, 1].cuda()
                labels = labels.cuda()

                preds = model(images1, images2)
                loss = loss_func(preds, labels)
                losses.append(loss.cpu().item())

                predictions = preds.cpu().numpy().reshape(-1)
                corrections = labels.cpu().numpy().reshape(-1)
                for p, c in zip(predictions, corrections):
                    num_sample += 1
                    if p >= 0.5 and c == 1:
                        num_correct += 1
                    elif p < 0.5 and c == 0:
                        num_correct += 1
                
            print("Verification loss : {:.5f}".format((np.array(losses).mean())))
            print("Verification accuracy : {:.3f}".format((float(num_correct / num_sample))))
            verification_losses.append(np.array(losses).mean())
            verification_acc.append(float(num_correct / num_sample))


    print("Learning Finished")
    print("Final Train Loss : {:.5f}".format((train_losses[-1])))
    print("Final Verification Loss: {:.5f}".format(verification_losses[-1]))
    print("Final Verification Accuracy: {:.5f}".format(verification_acc[-1]))


def main(args):
    torch.manual_seed(args.seed)
    model = Model()
    model.cuda()

    train_loader, verification_loader = preprocess(args.data_dir)

    train(model, train_loader, verification_loader, args.epochs, 
          args.train_batch_sz, args.verification_batch_sz, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/omniglot',
                        help='omniglot dataset directory path')
    parser.add_argument('--train_batch_sz', type=int, default=128,
                        help='train_batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='max epoch')
    parser.add_argument('--verification_batch_sz', type=int, default=1, 
                        help='verification batch size')
    parser.add_argument('--seed', type=int, default=0,
                        help='torch seed')
    parser.add_argument('--save_path', type=str, default='output/')
    args = parser.parse_args()

    main(args)