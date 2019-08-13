import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import os
import argparse

import siamese_net
import losses
import datasets


PATH = '.'
CKP_PATH = os.path.join(PATH, 'checkpoints')


def main(args):
    BATCH_SIZE = args.batch_size
    device = torch.device(args.device)

    # ---- load the data sets----
    train_data = datasets.Reddit_Img_Pair(args.data_path, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]), re_select=True)

    test_data = datasets.Reddit_Img_Pair(args.data_path, train=False, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]), re_select=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False)
    data_loader = {'train_dataloader': train_loader,
                   'test_dataloader': test_loader}

    # ---- initialize the model----
    model = siamese_net.SiameseNet(model_type=args.model_type).to(device)

    train(data_loader, model, args.epochs, BATCH_SIZE,
          device, args.check_point, args.save_dir)


def train(data_loader, model, epoch, batch_size, device, checkpoint, save_dir):
    # ---- optimizer and loss function----
    # optimizer = optim.SGD(model.parameters(), lr=0.01)  # can also use Adam
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = losses.PairCrossEntropy(size_average=True)
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = data_loader['train_dataloader']

    os.makedirs(CKP_PATH, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    # # add the function of checkpoint
    epoch_ckp = 0
    if not checkpoint == 0:

        print('  start from{} checkpoint \n'.format(checkpoint))
        model_checkpoint = torch.load(
            os.join(PATH, 'checkpoints', '%d_model.pth' % (checkpoint)))

        model.load_state_dict(model_checkpoint['model_state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        epoch_ckp = model_checkpoint['epoch']
        loss = model_checkpoint['loss']

    model.train()
    for epoch_i in range(epoch_ckp, epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = [data_i.to(device)
                            for data_i in data], target.to(device)

            optimizer.zero_grad()
            output1, output2 = model(data)
            loss = criterion(output1, output2, target)
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_i, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        # ----save the checkpoint ----
        torch.save({
            'epoch': epoch_i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.join(PATH, 'checkpoints', '%d_model.pth' % (epoch_i)))

        # ---- test after each epoch----
        test_loader = data_loader['test_dataloader']
        test(model, test_loader, device)

        # save the model


def test(model, test_loader, device):
    with torch.no_grad():
        # it is encouraged to use the torch.no_grad to deactivate the auto-grad engine to speed up the computation
        model.eval()
        criterion = losses.PairCrossEntropy(size_average=False)
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = [data_i.to(device)
                            for data_i in data], target.to(device)
            output1, output2 = model(data)

            # sum up batch loss
            test_loss += criterion(output1, output2,
                                   target)

            # get the prediction results
            pred_prob = torch.squeeze(criterion.get_log_probability(
                output1, output2))
            pred = np.zeros(data[0].size()[0])
            pred[pred_prob.numpy() > 0.5] = 1

            result_array = np.zeros(data[0].size()[0])
            result_array[pred == target.numpy()] = 1
            correct += result_array.sum()

        # get the average ranking loss and accuracy
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train the siamese network')

    parser.add_argument('--data_path', type=str, default='.',
                        help='path of the project, the default is ./ ')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='the batch size fo the training, default is 64')
    parser.add_argument('--model_type', type=int, default=1,
                        help='the type of models for training.')
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
