##################################################################################################
# EEGNet with transfer learning for EEG signal classification.
# Author：Ce Ju, Dashan Gao
# Date  : July 29, 2020
# Paper : Ce Ju et al., Federated Transfer Learning for EEG Signal Classification, IEEE EMBS 2020.
##################################################################################################

import warnings
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from EEGNET import EEGNet
from EEGNET import Conv2dWithConstraint

warnings.filterwarnings('ignore')


def EEGNet_transfer_learning_experients(cov_data_good, labels_good, cov_data, labels, index, fold, subject):
    cov_data_train = np.concatenate((cov_data_good, cov_data[index != fold]), axis=0)
    """
    Conduct model training based on EEGNet
    :param cov_data: training data
    :param labels: training labels
    :param index:
    :param fold:
    :param subject: subject id
    :return:
    """
    cov_data_train = cov_data_train.reshape(
        (cov_data_train.shape[0], 1, cov_data_train.shape[1], cov_data_train.shape[2]))
    cov_data_test = cov_data[index == fold].reshape(
        (cov_data[index == fold].shape[0], 1, cov_data[index == fold].shape[1], cov_data[index == fold].shape[2]))

    input_data_train = Variable(torch.from_numpy(cov_data_train)).float()
    input_data_test = Variable(torch.from_numpy(cov_data_test)).float()
    target_train = Variable(torch.LongTensor(np.concatenate((labels_good, labels[index != fold]), axis=0)))
    target_test = Variable(torch.LongTensor(labels[index == fold]))

    model = EEGNet(n_chan=32, n_time=161)

    lr = 0.1
    old_loss = 1000
    iteration = 0

    while np.abs(old_loss) > 0.1:
        iteration += 1

        logits = model(input_data_train)
        output = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(output, target_train)
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target_train.data.view_as(pred)).long().cpu().sum()

        model.zero_grad()
        loss.backward()
        if iteration % 10 == 0:
            print('Iteration {} loss: {:4f}'.format(iteration, loss.item()))
            lr = max(0.99 * lr, 0.005)

        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad

        if np.abs(loss.item() - old_loss) < 1e-6: break
        old_loss = loss.item()

    logits = model(input_data_test)
    output = F.log_softmax(logits, dim=-1)
    loss = F.nll_loss(output, target_test)
    pred = output.data.max(1, keepdim=True)[1]
    correct_test = pred.eq(target_test.data.view_as(pred)).long().cpu().sum()

    print('Subject {} (Fold {}) Classification Accuracy: {:4f}.'.format(subject, fold,
                                                                        correct_test.item() / pred.shape[0]))
    print('--------------------------------------')

    return correct_test.item() / pred.shape[0]


if __name__ == '__main__':

    np.random.seed(0)

    data = np.load('raw_data/normalized_original_epoch_data_train.npy')
    label = np.load('raw_data/train_label.npy')
    index = np.load('raw_data/index.npy')

    good_subject_index = [0, 1, 6, 7, 14, 28, 30, 32, 33, 34, 41, 47, 51, 53, 54, 55, 59, 61, 69, 70, 71, 72, 79, 84,
                          85, 92, 103]
    cov_data_good = np.concatenate(data[good_subject_index], axis=0)
    labels_good = np.concatenate(label[good_subject_index], axis=0)

    Total_average_acc = []

    # bad_subject_index = [2, 8, 16, 17, 22, 23, 27, 35, 37, 38, 39, 40, 44, 46, 57, 62, 63, 66, 73, 75, 76, 77, 89, 95, 96, 98, 100, 101]
    # for bad_subject_index in [2, 8, 16, 17, 22, 23, 27, 35, 37, 38, 39, 40, 44, 46, 57, 62, 63, 66, 73, 75, 76, 77, 89, 95, 96, 98, 100, 101]:
    for bad_subject_index in [16]:

        cov_data_subject = data[bad_subject_index]
        labels_subject = label[bad_subject_index]

        accuracy = []

        for fold in range(1, 6):
            for _ in range(1):
                accuracy.append(EEGNet_transfer_learning_experients(cov_data_good, labels_good, cov_data_subject, labels_subject,
                                                                    index[bad_subject_index], fold, bad_subject_index))

        print('--------------------------------')
        print(
            'Subject {}: Average Classification Accuracy: {:4f}.'.format(bad_subject_index, np.array(accuracy).mean()))
        Total_average_acc.append(np.array(accuracy).mean())

    print('-------------------------------------------------------------------')
    print('Total_Average_ACC: {:4f}'.format(np.mean(Total_average_acc)))
