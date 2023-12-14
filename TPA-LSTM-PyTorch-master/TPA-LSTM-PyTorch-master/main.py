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
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--skip', type=float, default=24)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='tanh')
args = parser.parse_args()

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = [0] * 6
    total_loss_l1 = [0] * 6
    n_samples = 0;
    # 初始化predict和test为长度为5的列表，每个元素都为None
    predict = [None] * 5
    Ytest = [None] * 5

    for X, Y in data.get_batches(X, Y, batch_size, False):
        with torch.no_grad():  # 确保在评估时不会计算梯度
            output = model(X)[:, :9]  # 获取模型前九个维度的输出
            if predict[0] is None:
                predict[0] = output;
                Ytest[0] = Y[:, 0, :9];
            else:
                predict[0] = torch.cat((predict[0], output));
                Ytest[0] = torch.cat((Ytest[0], Y[:, 0, :9]));
        scale = data.scale.expand(output.size(0), -1)[:, :9]  # 确保 scale 张量与 output 的批次大小匹配
        loss = evaluateL2(output * scale, Y[:, 0, :9] * scale)
        total_loss[0] += loss.item()  # 累加每一步的损失

        loss = evaluateL1(output * scale, Y[:, 0, :9] * scale)
        total_loss_l1[0] += loss.item()  # 累加每一步的损失
        for step in range(5 - 1):  # 已经有了1次初始预测，所以再做4次
            # 准备下一步的输入
            X_part = X[:, 1:, :]
            Y_part = Y[:, step, 9:]

            # 按最后一维拼接这两部分
            trick = torch.cat((output, Y_part), dim=1)
            trick = trick.unsqueeze(1)
            X = torch.cat((X_part, trick), dim=1)
            # 做出新的预测
            output = model(X)[:, :9]
            # 检查predict和test的当前元素是否为None，并进行赋值或拼接
            if predict[step+1] is None:
                predict[step+1] = output
                Ytest[step+1] = Y[:, step+1, :9]
            else:
                predict[step+1] = torch.cat((predict[step+1], output), dim=0)
                Ytest[step+1] = torch.cat((Ytest[step+1], Y[:, step+1, :9]), dim=0)

            # 累积损失
            loss = evaluateL2(output * scale, Y[:, step + 1, :9] * scale)
            total_loss[step+1] += loss.item()  # 累加每一步的损失
            loss = evaluateL1(output * scale, Y[:, step + 1, :9] * scale)
            total_loss_l1[step+1] += loss.item()  # 累加每一步的损失


        n_samples += (output.size(0) * output.size(1));

    avg_loss = sum(total_loss) / n_samples
    avg_loss_l1 = sum(total_loss_l1) / n_samples

    total_loss[-1] += avg_loss  # 累加平均损失
    total_loss_l1[-1] += avg_loss_l1  # 累加平均损失

    # 计算相对误差
    list_of_rse = [math.sqrt(loss_L2 / n_samples) / rse for loss_L2, rse in zip(total_loss, data.list_of_rse)]
    list_of_rae = [loss_L1 / n_samples / rae for loss_L1, rae in zip(total_loss_l1, data.list_of_rae)]

    # 假设 predict 和 Ytest 是长度为5的列表，每个元素都是一个Tensor
    predict = [p.detach().cpu().numpy() for p in predict if p is not None]
    Ytest = [y.detach().cpu().numpy() for y in Ytest if y is not None]

    # 现在 predict_tensors 和 Ytest_tensors 是包含numpy数组的列表

    #print(predict.shape, Ytest.shape)

    # 计算每个步骤的相关性
    correlations = []
    for step in range(5):
        if predict[step] is not None and Ytest[step] is not None:
            sigma_p = predict[step].std(axis=0)
            sigma_g = Ytest[step].std(axis=0)
            mean_p = predict[step].mean(axis=0)
            mean_g = Ytest[step].mean(axis=0)
            index = (sigma_g != 0)
            correlation = ((predict[step] - mean_p) * (Ytest[step] - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
            correlation = (correlation[index]).mean() if np.any(index) else 0
            correlations.append(correlation)
        else:
            correlations.append(None)
    mean_corr = sum(correlations) / len(correlations)
    correlations.append(mean_corr)
    return list_of_rse, list_of_rae, correlations;


def train(data, X, Y, model, criterion, optim, batch_size):  # X is train set, Y is validation set, data is the whole data
    model.train();
    total_losses = [0] * 6  # 初始化一个长度为6的列表，用于存储每步的损失和最后的平均损失
    n_samples = 0;
    for X, Y in data.get_batches(X, Y, batch_size, True):
        #print(Y)
        model.zero_grad();
        output = model(X)[:, :9]  # 获取模型前九个维度的输出
        # 切片 data.scale 以匹配前9个维度的输出
        scale = data.scale.expand(output.size(0), -1)[:, :9]  # 确保 scale 张量与 output 的批次大小匹配
        list_of_loss = []
        loss = criterion(output * scale, Y[:, 0, :9] * scale)
        list_of_loss.append(loss)
        total_losses[0] += loss.detach().item()  # 累加每一步的损失
        for step in range(5 - 1):  # 已经有了1次初始预测，所以再做4次
            # 准备下一步的输入
            X_part=X[:,1:,:]
            Y_part = Y[:, step, 9:]

            # 按最后一维拼接这两部分
            trick = torch.cat((output, Y_part), dim=1)
            trick=trick.unsqueeze(1)
            X=torch.cat((X_part, trick), dim=1)
            # 做出新的预测
            output = model(X)[:, :9]
            # 累积损失
            loss = criterion(output * scale, Y[:, step + 1, :9] * scale)
            list_of_loss.append(loss)
            total_losses[step+1] += loss.detach().item()  # 累加每一步的损失


        # 取5步预测的平均损失
        avg_loss = torch.stack(list_of_loss).mean()
        #avg_loss.backward();
        list_of_loss[0].backward();
        optim.step()
        total_losses[-1] += avg_loss.item()  # 累加平均损失
        n_samples += (output.size(0) * output.size(1))
        average_losses = [total_loss / n_samples for total_loss in total_losses]

    return average_losses
    return 1




Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize); #SPLITS THE DATA IN TRAIN AND VALIDATION SET, ALONG WITH OTHER THINGS, SEE CODE FOR MORE
print(Data.list_of_rse);

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
        val_loss, val_list_of_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size);
        (time.time() - epoch_start_time)
        # 假设 epoch 和 time_elapsed 已经是定义好的变量
        # 假设 train_loss, val_loss, val_list_of_rae, val_corr 都是列表

        print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)), end='')
        print('\n')  # 最后打印换行符

        # 使用 zip() 同时迭代所有列表
        for tl, vl, rae, corr in zip(train_loss, val_loss, val_list_of_rae, val_corr):
            print(
                'train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr {:5.4f} |'.format(tl, vl, rae,
                                                                                                           corr),
                end='\n ')
        print()  # 最后打印换行符

        # Save the model if the validation loss is the best we've seen so far.

        if val_loss[5].item() < best_val:
            # with open(args.save, 'wb') as f:
            #     torch.save(model, f)
            best_val = val_loss[5]
        if epoch % 5 == 0:
            test_acc, test_list_of_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                     args.batch_size);
            # 假设 test_acc, test_list_of_rae, test_corr 都是列表，并且它们的长度相同
            for acc, rae, corr in zip(test_acc, test_list_of_rae, test_corr):
                print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(acc, rae, corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')