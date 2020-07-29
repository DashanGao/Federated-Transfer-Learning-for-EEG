##################################################################################################
# Train EEG signal classifier using EEGNet as baseline.
# Author：Ce Ju, Dashan Gao
# Date  : July 29, 2020
# Paper : Ce Ju et al., Federated Transfer Learning for EEG Signal Classification, IEEE EMBS 2020.
##################################################################################################

import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')
np.random.seed(0)


class EEGNet(nn.Module):
    '''
    EEGNet architecture:
        The expected input during runtime is in a formate:
            miniBatch x 1 x electroldes x time points
        During initialization set:
            nChan : number of electrodes
            nTime : number of samples
            nClasses: number of classes
    '''

    def __init__(self, n_chan, n_time, n_class=2,
                 dropoutP=0.25, F1=4, D=2,
                 C1=100, *args, **kwargs):
        super(EEGNet, self).__init__()
        self.F2 = D * F1
        self.F1 = F1
        self.D = D
        self.n_time = n_time
        self.n_class = n_class
        self.n_chan = n_chan
        self.C1 = C1

        self.first_blocks = self.initial_blocks(dropoutP)
        self.f_size = self.calculateOutSize(self.first_blocks, n_chan, n_time)
        self.last_layer = self.lastBlock(self.F2, n_class, (1, self.f_size[1]))

    def forward(self, x):
        x = self.first_blocks(x)
        x = self.last_layer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        return x

    def initial_blocks(self, dropoutP, *args, **kwargs):
        block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.C1),
                      padding=(0, self.C1 // 2), bias=False),
            nn.BatchNorm2d(self.F1),
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.n_chan, 1),
                                 padding=0, bias=False, max_norm=1,
                                 groups=self.F1),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropoutP))
        block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 22),
                      padding=(0, 22 // 2), bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                      stride=1, bias=False, padding=0),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutP)
        )
        return nn.Sequential(block1, block2)

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(inF, outF, kernalSize, *args, **kwargs),
            nn.LogSoftmax(dim=1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


def EEGNet_local_training_experients(cov_data, labels, index, fold, subject):
    """
    Conduct model training based on EEGNet
    :param cov_data: training data
    :param labels: training labels
    :param index:
    :param fold:
    :param subject: subject id
    :return:
    """
    cov_data_train = cov_data[index != fold].reshape(
        (cov_data[index != fold].shape[0], 1, cov_data[index != fold].shape[1], cov_data[index != fold].shape[2]))
    cov_data_test = cov_data[index == fold].reshape(
        (cov_data[index == fold].shape[0], 1, cov_data[index == fold].shape[1], cov_data[index == fold].shape[2]))

    input_data_train = Variable(torch.from_numpy(cov_data_train)).float()
    input_data_test = Variable(torch.from_numpy(cov_data_test)).float()
    target_train = Variable(torch.LongTensor(labels[index != fold]))
    target_test = Variable(torch.LongTensor(labels[index == fold]))

    model = EEGNet(nChan=32, nTime=161)

    lr = 0.1
    old_loss = 1000
    iteration = 0

    while np.abs(old_loss) > 0.01:
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

    data = np.load('raw_data/normalized_original_epoch_data_train.npy')
    label = np.load('raw_data/train_label.npy')
    index = np.load('index.npy')

    accuracy = []
    all_accuracy = []

    for subject in range(100, 108):
        accuracy = []
        for fold in range(1, 6):
            cov_data_subject = data[subject]
            label_subject = label[subject]
            accuracy.append(EEGNet_local_training_experients(cov_data_subject, label_subject, index[subject], fold, subject))
        all_accuracy.append(np.mean(np.array(accuracy)))
        print('Subject {} Average {:6f}'.format(subject, np.mean(np.array(accuracy))))

    print('All Accuracy: ', all_accuracy)
    print('Average Classification Accuracy: {:4f}.'.format(np.array(all_accuracy).mean()))
