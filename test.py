import os
import sys
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from UARmodel import UARmodel
from UARdataset import UARdataset


parser = argparse.ArgumentParser()
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--seed', type=int,  default=1337)
parser.add_argument('--gpu', type=str,  default='0')
args = parser.parse_args()

if args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

model = UARmodel()
model.cuda()
model.load_state_dict(torch.load('./checkpoints_MUAR/best_mse.pth'))
model.eval()


def MAPE(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true)).item()


def MSE(y_true, y_pred):
    return torch.mean(torch.square(y_true - y_pred)).item()


def MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()


test_data = np.load('./SWellEx-96-S5/test_data_1.npy')
test_label = pd.read_csv('./SWellEx-96-S5/test_label_1.csv')


if __name__ == '__main__':

    snapshot_path = './prediction_MUAR/'
    if not os.path.exists(snapshot_path):
        os.mkdir(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    testdata = UARdataset(test_data, test_label)
    testloader = DataLoader(testdata, batch_size=16, shuffle=False)

    running_mse = 0
    running_mape = 0
    running_mae = 0

    with torch.no_grad():
        for i, (inputs, label) in enumerate(testloader):
            inputs, label = inputs.float().cuda(), label.float().cuda()
            output, F_p4, F_p3, F_p2 = model(inputs)
            output, F_p4, F_p3, F_p2 = output.squeeze(), F_p4.squeeze(), F_p3.squeeze(), F_p2.squeeze()
            final_output = (output + F_p4 + F_p3 + F_p2) / 4

            mse = MSE(label, final_output)
            running_mse += mse
            mape = MAPE(label, final_output)
            running_mape += mape
            mae = MAE(label, final_output)
            running_mae += mae

    epoch_mse = running_mse / len(testloader)
    epoch_mape = running_mape / len(testloader)
    epoch_mae = running_mae / len(testloader)

    logging.info('MSE : %f  MAPE : %f  MAE : %f' % (epoch_mse, epoch_mape, epoch_mae))
