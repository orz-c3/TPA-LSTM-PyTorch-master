"""THE DRIVER CLASS TO RUN THIS CODE"""

"""FUTURE SCOPE, ADD ARGUMENTS AS NEEDED"""


import argparse
import math
import time

import torch
import torch.nn as nn

from models import TPA_LSTM_Modified
from utils import *
import numpy as np;
import importlib

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default="data/jl_data_train.csv",
                    help='location of the data file')
#, required=True
parser.add_argument('--model', type=str, default='TPA_LSTM_Modified',
                    help='')
parser.add_argument('--hidden_state_features', type=int, default=12,
                    help='number of features in LSTMs hidden states')
parser.add_argument('--num_layers_lstm', type=int, default=1,
                    help='num of lstm layers')
parser.add_argument('--hidden_state_features_uni_lstm', type=int, default=1,
                    help='number of features in LSTMs hidden states for univariate time series')
parser.add_argument('--num_layers_uni_lstm', type=int, default=1,
                    help='num of lstm layers for univariate time series')
parser.add_argument('--attention_size_uni_lstm', type=int, default=10,
                    help='attention size for univariate lstm')
parser.add_argument('--hidCNN', type=int, default=10,
                    help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=100,
                    help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=30,
                    help='window size')
parser.add_argument('--CNN_kernel', type=int, default=1,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=5,
                    help='The window size of the highway component')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=3000,
                    help='upper epoch limit') #30
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=False)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=1e-05)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--skip', type=float, default=24)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='tanh')
args = parser.parse_args()

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    for X, Y in data.get_batches(X, Y, batch_size, False):
        with torch.no_grad():  # 确保在评估时不会计算梯度
            output = model(X)[:, :9]  # 获取模型前九个维度的输出
            Y = Y[:, :9]  # 确保目标也只有九个维度
            if predict is None:
                predict = output;
                test = Y;
            else:
                predict = torch.cat((predict, output));
                test = torch.cat((test, Y));



        # 循环预测的剩余部分
        for step in range(1, 5):
        # 更新输入，将预测的输出置于最后一位置
            X = torch.roll(X, -1, dims=1)
            X[:, -1, :9] = output
            output = model(X)[:, :9]
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

            # 计算损失
            scale = data.scale.expand(output.size(0), -1)[:, :9]  # 确保 scale 张量与 output 的批次大小匹配

            total_loss += evaluateL2(output * scale, Y * scale).data
            total_loss_l1 += evaluateL1(output * scale, Y * scale).data
            n_samples += (output.size(0) * output.size(1));

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();

    #print(predict.shape, Ytest.shape)

    sigma_p = (predict).std(axis=0);
    sigma_g = (Ytest).std(axis=0);
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    return rse, rae, correlation;


def train(data, X, Y, model, criterion, optim, batch_size):  # X is train set, Y is validation set, data is the whole data
    model.train();
    total_loss = 0;
    n_samples = 0;
    for X, Y in data.get_batches(X, Y, batch_size, True):
        #print(Y)
        model.zero_grad();
        output = model(X)[:, :9]  # 获取模型前九个维度的输出
        Y = Y[:, :9]  # 确保目标也只有九个维度

        # 切片 data.scale 以匹配前9个维度的输出
        scale = data.scale[:, :9] if len(data.scale.shape) > 1 else data.scale[:9]
        scale = scale.expand(output.size(0), 9)  # 确保 scale 张量与 output 的批次大小匹配

        loss = criterion(output * scale, Y * scale);
        # 进行5次循环预测
        for step in range(5 - 1):  # 已经有了1次初始预测，所以再做4次
            # 准备下一步的输入
            X = torch.roll(X, -1, dims=1)  # 将输入向左滚动
            X[:, -1, :9] = output.detach()  # 使用前一次预测的输出作为新输入的最后一部分
            # 做出新的预测
            output = model(X)[:, :9]
            # 累积损失
            loss += criterion(output * scale, Y * scale);

        loss /= 5  # 取5步预测的平均损失
        loss.backward();
        grad_norm = optim.step();
        total_loss += loss.data;
        n_samples += (output.size(0) * output.size(1));
    return total_loss / n_samples
    return 1




Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize); #SPLITS THE DATA IN TRAIN AND VALIDATION SET, ALONG WITH OTHER THINGS, SEE CODE FOR MORE
print(Data.rse);

device = 'cpu'


model = eval(args.model).Model(args, Data);
if(args.cuda):
    model.cuda()


#print(dict(model.named_parameters()))
if args.L1Loss:
    criterion = nn.L1Loss(size_average=False);
else:
    criterion = nn.MSELoss(size_average=False);
evaluateL2 = nn.MSELoss(size_average=False);
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

#print(list(model.parameters())[0].grad)
list(model.parameters())
#optim = Optim.Optim(model.parameters(), args.optim, args.lr, args.clip,)
# 根据命令行参数选择优化器
if args.optim.lower() == 'adam':
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim.lower() == 'sgd':
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
else:
    raise ValueError("Optimization method not supported: {}".format(args.optim))

best_val = 10000000;

try:
    print('begin training');
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
        print(train_loss)
        val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size);
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.

        if val_loss < best_val:
            # with open(args.save, 'wb') as f:
            #     torch.save(model, f)
            best_val = val_loss
        if epoch % 5 == 0:
            test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                     args.batch_size);
            print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')