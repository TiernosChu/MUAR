import os
import sys
import copy
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from UARmodel import UARmodel
from UARdataset import UARdataset


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', default=0.001)
parser.add_argument('--weight_decay', default=0.00005)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))

if args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def MAPE(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true)).item()


def MSE(y_true, y_pred):
    return torch.mean(torch.square(y_true - y_pred)).item()


def MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()


def huber(true, pred, delta=1.0):
    loss = torch.where(torch.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2), delta * torch.abs(true - pred) - 0.5 * (delta ** 2))
    return torch.mean(loss)


model = UARmodel()

train_data = np.load('./SWellEx-96-S5/train_data_1.npy')
train_label = pd.read_csv('./SWellEx-96-S5/train_label_1.csv')
valid_data = np.load('./SWellEx-96-S5/valid_data_1.npy')
valid_label = pd.read_csv('./SWellEx-96-S5/valid_label_1.csv')

if __name__ == '__main__':

    snapshot_path = './checkpoints_MUAR/'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model.cuda()

    Traindata = UARdataset(train_data, train_label)
    Trainloader = DataLoader(Traindata, batch_size=batch_size, shuffle=True)

    validdata = UARdataset(valid_data, valid_label)
    validloader = DataLoader(validdata, batch_size=batch_size, shuffle=False)

    dataset = {'train': Trainloader, 'valid': validloader}
    dataset_sizes = {'train': len(Trainloader), 'valid': len(validloader)}

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    writer = SummaryWriter(snapshot_path + 'log')

    best_model_wts_1 = copy.deepcopy(model.state_dict())
    best_model_wts_2 = copy.deepcopy(model.state_dict())
    best_model_wts_3 = copy.deepcopy(model.state_dict())
    best_model = copy.deepcopy(model.state_dict())
    best_mse = float('inf')
    best_mape = float('inf')
    best_mae = float('inf')
    epoch1, epoch2, epoch3 = 0, 0, 0
    num_epochs = args.num_epochs

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_mse = 0
            running_mape = 0
            running_mae = 0

            for i, (inputs, label) in enumerate(dataset[phase]):
                inputs, label = inputs.float().cuda(), label.float().cuda()

                with torch.set_grad_enabled(phase == 'train'):
                    output, F_p4, F_p3, F_p2 = model(inputs)
                    output, F_p4, F_p3, F_p2 = output.squeeze(), F_p4.squeeze(), F_p3.squeeze(), F_p2.squeeze()
                    loss_output = huber(output, label)
                    loss_p4 = huber(F_p4, label)
                    loss_p3 = huber(F_p3, label)
                    loss_p2 = huber(F_p2, label)
                    loss = loss_output + loss_p4 + loss_p3 + loss_p2
                    final_output = (output + F_p4 + F_p3 + F_p2) / 4

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step(epoch + i / len(Trainloader))

                running_loss += loss.item()
                mse = MSE(label, final_output)
                running_mse += mse
                mape = MAPE(label, final_output)
                running_mape += mape
                mae = MAE(label, final_output)
                running_mae += mae

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_mse = running_mse / dataset_sizes[phase]
            epoch_mape = running_mape / dataset_sizes[phase]
            epoch_mae = running_mae / dataset_sizes[phase]

            logging.info(phase + '_epoch %d : loss : %f  MSE : %f  MAPE : %f  MAE : %f' % (
                epoch + 1, epoch_loss, epoch_mse, epoch_mape, epoch_mae))

            writer.add_scalar(phase + '_loss/loss', epoch_loss, epoch + 1)
            writer.add_scalar(phase + '_MSE', epoch_mse, epoch + 1)
            writer.add_scalar(phase + '_MAPE', epoch_mape, epoch + 1)
            writer.add_scalar(phase + '_MAE', epoch_mae, epoch + 1)

            if phase == 'valid' and epoch_mse < best_mse:
                best_mse = epoch_mse
                epoch1 = epoch + 1
                best_model_wts_1 = copy.deepcopy(model.state_dict())

            if phase == 'valid' and epoch_mape < best_mape:
                best_mape = epoch_mape
                epoch2 = epoch + 1
                best_model_wts_2 = copy.deepcopy(model.state_dict())

            if phase == 'valid' and epoch_mae < best_mae:
                best_mae = epoch_mae
                epoch3 = epoch + 1
                best_model_wts_3 = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts_1)
    torch.save(model.state_dict(), snapshot_path + '/best_mse.pth')

    model.load_state_dict(best_model_wts_2)
    torch.save(model.state_dict(), snapshot_path + '/best_mape.pth')

    model.load_state_dict(best_model_wts_3)
    torch.save(model.state_dict(), snapshot_path + '/best_mae.pth')

    best_model = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), snapshot_path + '/final_epoch.pth')

    logging.info('Best_mse : %f  Best_mape : %f  Best_mae : %f' % (best_mse, best_mape, best_mae))
    logging.info('best_mse_epoch : %f  best_mape_epoch : %f  best_mae_epoch : %f' % (epoch1, epoch2, epoch3))
