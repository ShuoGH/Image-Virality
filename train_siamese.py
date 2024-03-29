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

import siamese_net
import losses
import datasets
import utils


PATH = '.'
CKP_PATH = os.path.join(PATH, 'checkpoints')


def main(args):
    BATCH_SIZE = args.batch_size
    device = torch.device(args.device)

    # ---- load the data sets----
    train_data = datasets.Reddit_Img_Pair(args.data_path, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]), re_select=True, pair_mode=args.pair_mode)

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
    # the model type: 1 for the normal viral net, 2 for the model combining the global image
    # Device: avoid the problem of not in the same device
    # freeze_* flag: whether freeze certain the part of the viral net.
    model = siamese_net.SiameseNet(
        model_type=args.model_type, model_name=args.model_name, device=device, freeze_locnet=args.freeze_locnet, freeze_ranknet=args.freeze_ranknet).to(device)

    train(data_loader, model, args.epochs, BATCH_SIZE,
          device, args.check_point, args.save_dir)


def train(data_loader, model, epoch, batch_size, device, checkpoint, save_dir):
    # ---- optimizer and loss function----
    # optimizer = optim.SGD(model.parameters(), lr=0.01)  # can also use Adam
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=0.001)
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
            os.path.join(PATH, 'checkpoints', '%d_model.pth' % (checkpoint)))

        model.load_state_dict(model_checkpoint['model_state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        epoch_ckp = model_checkpoint['epoch']
        loss = model_checkpoint['loss']

    model.train()

    train_loss_batch_list = []
    temp_epoch_list = []
    train_loss_list = []

    test_loss_list = []
    test_accuracy_list = []

    # ---- train for the epoch----
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
                epoch_i, batch_idx *
                data[0].size()[0], len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_loss_batch_list.append(loss.item())
            temp_epoch_list.append(loss.item())

        train_loss_list.append(np.mean(temp_epoch_list))

        # # ----save the checkpoint ----
        # If you don't have that much space, just ignore this part.
        # if epoch_i % 10 == 0:
        #     torch.save({
        #         'epoch': epoch_i,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': loss,
        #     }, os.path.join(PATH, 'checkpoints', '%d_model.pth' % (epoch_i)))

        # ---- test after each epoch----
        test_loader = data_loader['test_dataloader']
        test_loss, test_accuracy = test(model, test_loader, device)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)

    # save the model after the training
    torch.save({
        'epoch': epoch_i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(PATH, args.save_dir, 'model_floc{}_frank{}_.pth'.format(args.freeze_locnet, args.freeze_ranknet)))

    # save the records to ouput file
    x = np.array(train_loss_batch_list)
    y = np.array(train_loss_list)
    z = np.array(test_accuracy_list)
    k = np.array(test_loss_list)

    np.savetxt('./save/train_loss_batch_{}_{}_{}'.format(
        args.freeze_locnet, args.freeze_ranknet, args.pair_mode), x)
    np.savetxt('./save/train_loss_epoch_{}_{}_{}'.format(
        args.freeze_locnet, args.freeze_ranknet, args.pair_mode), y)
    np.savetxt('./save/test_loss_{}_{}_{}'.format(
        args.freeze_locnet, args.freeze_ranknet, args.pair_mode), k)
    np.savetxt('./save/test_accuracy_{}_{}_{}'.format(
        args.freeze_locnet, args.freeze_ranknet, args.pair_mode), z)


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
                output1, output2)).cpu()
            pred = np.zeros(data[0].size()[0])
            pred[pred_prob.numpy() > 0.5] = 1

            result_array = np.zeros(data[0].size()[0])
            result_array[pred == target.cpu().numpy()] = 1
            correct += result_array.sum()

        # get the average ranking loss and accuracy
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train the siamese network')

    parser.add_argument('--data_path', type=str, default='.',
                        help='path of the project, the default is ./ ')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='the batch size fo the training, default is 64')
    parser.add_argument('--model_name', type=str, default='alexnet',
                        help='the base model embedded in the siamese network, alexnet or densenet.')

    parser.add_argument('--model_type', type=int, default=1,
                        help='the type of models for training.')
    parser.add_argument('--pair_mode', type=int, default=1,
                        help='the type of dataset selected for training. Choose from 1,2,3,4')

    # parser.add_argument('--freeze_pretrained', type=int, default=0,
    #                     help='the flag which indicates whether freeze the pretrained model(like AlexNet) during training.')

    parser.add_argument('--freeze_locnet', type=int, default=0,
                        help='flag which indicates whether freeze the pretrained part of the locnet')
    parser.add_argument('--freeze_ranknet', type=int, default=0,
                        help='flag which indicates whether freeze the pretrained part of the ranknet')

    parser.add_argument('--device', type=str, default='cpu',
                        help='the device used for training.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='the number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='the initial learning rate.')
    parser.add_argument('--check_point', type=int, default=0,
                        help='the checkpoint of the training, 0 indicates training from scratch.')

    parser.add_argument('--save_dir', type=str, default='save',
                        help='the dir of the output model.')
    args = parser.parse_args()

    main(args)
