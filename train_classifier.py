import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, models
import numpy as np
import logging
import os
import argparse

import classifier_net
import losses
import datasets
import utils


PATH = '.'
CKP_PATH = os.path.join(PATH, 'checkpoints')


def main(args):
    BATCH_SIZE = args.batch_size
    # ---- load the data ----
    train_data = datasets.Reddit_Images_for_Classification('.', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]), re_split=True, balance=args.balance_data)
    test_data = datasets.Reddit_Images_for_Classification('.', train=False, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]), re_split=True, balance=args.balance_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False)
    data_loader = {'train_dataloader': train_loader,
                   'test_dataloader': test_loader}

    # ----intialize the model and do the freeze if needed----
    model = classifier_net.cnn_classifier(
        model_name=args.model, num_classes=5).to(args.device)
    if args.freeze_features:
        model.freeze()

    # ----begin the training----
    train(data_loader, model, args.epochs,
          BATCH_SIZE, args.device, args.check_point)


def train(data_loader, model, epoch, batch_size, device, checkpoint):

    # ---- optimizer and loss function----
    # optimizer = optim.SGD(model.parameters(), lr=0.01)  # can also use Adam
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = data_loader['train_dataloader']

    os.makedirs(CKP_PATH, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    # # add the function of checkpoint
    epoch_ckp = 0
    if not checkpoint == 0:

        print('  training the classifier from the {} checkpoint \n'.format(checkpoint))
        model_checkpoint = torch.load(
            os.path.join(PATH, 'checkpoints', '%d_classifier_model.pth' % (checkpoint)))

        model.load_state_dict(model_checkpoint['model_state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        epoch_ckp = model_checkpoint['epoch']
        loss = model_checkpoint['loss']

    model.train()
    # ---- train for the epoch----
    for epoch_i in range(epoch_ckp, epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # by default, the crossentyopy calculate the average loss of the images
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_i, batch_idx *
                BATCH_SIZE, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        # ----save the checkpoint ----
        if epoch_i % 5 == 0:
            torch.save({
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(PATH, 'checkpoints', '%d_classifier_model.pth' % (epoch_i)))

        # ---- test after each epoch----
        test_loader = data_loader['test_dataloader']
        test(model, test_loader, device)


def test(model, test_loader, device):
    with torch.no_grad():
        # it is encouraged to use the torch.no_grad to deactivate the auto-grad engine to speed up the computation
        model.eval()
        criterion = nn.CrossEntropyLoss()

        total_test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)  # the size of output is (64,5)

            # sum up batch loss
            total_test_loss += criterion(output, target)

            # get the prediction results
            _, predicted = torch.max(output, 1)
            total += target.size()[0]
            correct += (predicted == labels).sum().item()

        # get the average loss and the accuracy
        test_loss = total_test_loss / (batch_idx + 1)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train the classifier for predicting the subreddit of image')

    parser.add_argument('--data_path', type=str, default='.',
                        help='path of the project, the default is ./ ')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='the batch size fo the training, default is 64')
    parser.add_argument('--balance_data', type=int, default=1,
                        help='indicate whether to use the balanced data or not ')

    parser.add_argument('--model', type=str, default='vgg',
                        help='the type of model used as the classifier.')
    parser.add_argument('--pretrained', type=int, default=1,
                        help='indicate whether to use the pretrained model')
    parser.add_argument('--freeze_features', type=int, default=0,
                        help='indicate whether to freeze the features layers in the model. 1: True, 0: False.')

    parser.add_argument('--device', type=str, default='cpu',
                        help='the device used for training.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='the number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='the initial learning rate.')
    parser.add_argument('--check_point', type=int, default=0,
                        help='the checkpoint of the training, 0 indicates training from scratch.')

    parser.add_argument('--save_dir', type=str, default='./save/',
                        help='the dir of the output model.')
    args = parser.parse_args()

    main(args)
