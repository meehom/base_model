
from data_toolkit.hyperparameters import HyperParametersBase
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch.nn as nn
import timeit
import os
import pandas as pd
from basic_network import MLP
import sys
sys.path.append(".")


def main(data_set_directory):
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = np.load(data_set_directory, allow_pickle=True)
    training_cases_matrix = np.array(train_data, dtype=np.float64)
    training_cases_matrix = [torch.from_numpy(
        case).float() for case in training_cases_matrix]

    training_cases = training_cases_matrix[:100]   # 划分训练集和验证集
    validation_cases = training_cases_matrix[100:]

    feature_x = 4
    feature_y = 1
    full_case_count = len(train_data)
    hyper_parameters = HyperParametersBase(
        full_case_count, feature_x, feature_y)

    model = MLP(hyper_parameters.feature_num, hyper_parameters.target_num)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyper_parameters.initial_lr)
    # 一次一变学习率，学习率衰减为lr_decay_rate
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=hyper_parameters.lr_decay_rate)
    loss_function = nn.MSELoss()
    current_epoch = 0
    train_loss_epoch = []  # 画train的loss图
    dev_loss_epoch = []  # 画dev的loss图
    best_dev_loss = np.Inf  # 设置最优的测试集loss是无穷大

    # training_cases = []
    # validation_cases = []

    # 这里还可以修改为k折交叉验证
    training_cases = torch.utils.data.DataLoader(
        training_cases, batch_size=hyper_parameters.batch_size, shuffle=True)
    validation_cases = torch.utils.data.DataLoader(
        validation_cases, batch_size=len(validation_cases), shuffle=False)

    # train
    while current_epoch < hyper_parameters.num_epochs + 1:
        start_time = timeit.default_timer()
        print('Epoch:', current_epoch)
        train_loss_batches = []
        dev_loss_batches = []

        for batch_id, batch_data in enumerate(training_cases):
            optimizer.zero_grad()

            x = batch_data[:, :hyper_parameters.feature_num]
            predictions = model(x)
            # print(predictions)
            ground_truth = batch_data[:,
                                      hyper_parameters.feature_num:]
            train_loss = loss_function(predictions, ground_truth)
            train_loss.backward()
            optimizer.step()
            train_loss_batches.append(train_loss.item())

        scheduler.step()  # 1个epoch学习率衰减一下
        train_loss_epoch.append(
            np.mean(train_loss_batches))  # 最后记录所有epoch的loss

        # 在验证集测试, 反向传播不在求导了
        with torch.no_grad():
            for batch_id, batch_data in enumerate(validation_cases):
                x = batch_data[:, :hyper_parameters.feature_num]
                predictions = model(x)
                ground_truth = batch_data[:,
                                          hyper_parameters.feature_num:]

                dev_loss = loss_function(predictions, ground_truth)
                dev_loss_batches.append(dev_loss.item())
            dev_loss_epoch.append(np.mean(dev_loss_batches))
            is_best = False
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                is_best = True

             # save model
            if is_best:
                best_model_dir = 'result/basic_model/saved_models/'+str(seed)
                f_path = best_model_dir + '/best.pt'
                checkpoint = {'epoch': current_epoch, 'state_dict': model.state_dict(
                ), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler}
                torch.save(checkpoint, f_path)
            elapsed = timeit.default_timer() - start_time
            print('Time taken:', elapsed, 'loss:', np.mean(train_loss_batches))
            current_epoch += 1


if __name__ == '__main__':
    # data_set_dir = "data_batching/clean_data/trainingSet.npy"
    data_set_dir = sys.argv[1]
    main(data_set_dir)
