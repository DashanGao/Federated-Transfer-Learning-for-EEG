##################################################################################################
# FTL Draft Code for Subject-adaptive Analysis
# Author：CE JU
# Date  : April 20, 2020
# Paper : Ce Ju et al., Federated Transfer Learning for EEG Signal Classification, IEEE EMBS 2020.
# Description: Source domain inlcudes all good subjects, target domain is the bad subject.
##################################################################################################

import numpy as np
import model
from MMD_loss import MMD_loss
import matplotlib
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('Agg')


def transfer_SPD(cov_data_1, cov_data_2, labels_1, labels_2):
    np.random.seed(0)
    ##First Subject
    random_index_1 = np.arange(cov_data_1.shape[0])

    cov_data_1 = cov_data_1[random_index_1, :, :]
    labels_1 = labels_1[random_index_1]

    split_num_1 = cov_data_1.shape[0]
    cov_data_train_1 = cov_data_1[0:split_num_1, :, :]

    ##Second Subject
    random_index_2 = np.arange(cov_data_2.shape[0])

    cov_data_2 = cov_data_2[random_index_2, :, :]
    labels_2 = labels_2[random_index_2]

    split_num_2 = int(np.floor(cov_data_2.shape[0] * 0.8))
    cov_data_train_2 = cov_data_2[0:split_num_2, :, :]
    cov_data_test_2 = cov_data_2[split_num_2:cov_data_2.shape[0], :, :]

    print('split_num_for_test: ', split_num_2)
    print('rest_num_for_test: ', labels_2.shape[0] - split_num_2)
    print('-------------------------------------------------------')

    input_data_train_1 = Variable(torch.from_numpy(cov_data_train_1)).double()
    input_data_train_2 = Variable(torch.from_numpy(cov_data_train_2)).double()
    input_data_test_2 = Variable(torch.from_numpy(cov_data_test_2)).double()

    target_train_1 = Variable(torch.LongTensor(labels_1[0:split_num_1]))
    target_test_1 = Variable(torch.LongTensor(labels_1[split_num_1:labels_1.shape[0]]))

    target_train_2 = Variable(torch.LongTensor(labels_2[0:split_num_2]))
    target_test_2 = Variable(torch.LongTensor(labels_2[split_num_2:labels_2.shape[0]]))

    model_1 = model.SPDNetwork_1()
    model_2 = model.SPDNetwork_2()

    old_loss = 0

    for iteration in range(200):

        logits_1, feat_1 = model_1(input_data_train_1)
        logits_2, feat_2 = model_2(input_data_train_2)

        index_positive_1 = np.array(target_train_1) == 1
        index_negative_1 = np.array(target_train_1) == 0

        index_positive_2 = np.array(target_train_2) == 1
        index_negative_2 = np.array(target_train_2) == 0

        feat_1_positive = feat_1[index_positive_1].detach().numpy()
        feat_1_negative = feat_1[index_negative_1].detach().numpy()
        feat_2_positive = feat_2[index_positive_2].detach().numpy()
        feat_2_negative = feat_2[index_negative_2].detach().numpy()

        feat_1_positive = Variable(torch.from_numpy(feat_1_positive)).double()
        feat_2_positive = Variable(torch.from_numpy(feat_2_positive)).double()

        feat_1_negative = Variable(torch.from_numpy(feat_1_negative)).double()
        feat_2_negative = Variable(torch.from_numpy(feat_2_negative)).double()

        output_1 = F.log_softmax(logits_1, dim=-1)
        output_2 = F.log_softmax(logits_2, dim=-1)

        pred_1 = output_1.data.max(1, keepdim=True)[1]
        pred_2 = output_2.data.max(1, keepdim=True)[1]

        mmd = MMD_loss('rbf', kernel_mul=2.0)

        loss = F.nll_loss(output_1, target_train_1) + F.nll_loss(output_2, target_train_2) + \
               1 * mmd.forward(feat_1_positive, feat_2_positive) + \
               1 * mmd.forward(feat_1_negative, feat_2_negative)

        correct_1 = pred_1.eq(target_train_1.data.view_as(pred_1)).long().cpu().sum()
        correct_2 = pred_2.eq(target_train_2.data.view_as(pred_2)).long().cpu().sum()

        loss.backward()

        lr_1, lr_2 = 0.1, 0.1
        grad_1 = model_1.update_para(lr_1)
        grad_2 = model_2.update_para(lr_2)

        average_grad = (grad_1 + grad_2) / 2

        lr = 0.1
        model_1.second_update_para(lr, average_grad)
        model_2.second_update_para(lr, average_grad)

        if iteration % 50 == 0:
            lr = max(0.98 * lr, 0.01)
            lr_1 = max(0.98 * lr_1, 0.01)
            lr_2 = max(0.98 * lr_2, 0.01)

        print('Iteration {}: Trainning Accuracy for Model 1: {:.4f} / Model 2: {:.4f}'.format(iteration,
                                                                                              correct_1.item() /
                                                                                              pred_1.shape[0],
                                                                                              correct_2.item() /
                                                                                              pred_2.shape[0]))

        logits_2, _ = model_2(input_data_test_2)
        output_2 = F.log_softmax(logits_2, dim=-1)
        loss_2 = F.nll_loss(output_2, target_test_2)
        pred_2 = output_2.data.max(1, keepdim=True)[1]
        correct_test_2 = pred_2.eq(target_test_2.data.view_as(pred_2)).long().cpu().sum()
        print('Testing Accuracy for Model 2: {:.4f}'.format(correct_test_2.item() / pred_2.shape[0]))

        if np.abs(loss.item() - old_loss) < 1e-4: break
        old_loss = loss.item()

    return correct_test_2.item() / pred_2.shape[0]


if __name__ == '__main__':

    np.random.seed(0)

    data = np.load('raw_data/normalized_original_train_sample.npy')
    label = np.load('raw_data/train_label.npy')

    # Good Subject
    cov_data_1 = np.concatenate(data[
                                    [0, 1, 6, 7, 14, 28, 30, 32, 33, 34, 41, 47, 51, 53, 54, 55, 59, 61, 69, 70, 71, 72,
                                     79, 84, 85, 92, 99, 103]], axis=0)
    labels_1 = np.concatenate(label[[0, 1, 6, 7, 14, 28, 30, 32, 33, 34, 41, 47, 51, 53, 54, 55, 59, 61, 69, 70, 71, 72,
                                     79, 84, 85, 92, 99, 103]], axis=0)

    # Bad Subject
    cov_data_2 = data[100]
    labels_2 = label[100]

    accuracy = []

    for _ in range(10):
        accuracy.append(transfer_SPD(cov_data_1, cov_data_2, labels_1, labels_2))

    print('All Accuracy: ', accuracy)
    print('SPD Riemannian Average Classification Accuracy: {:4f}.'.format(np.array(accuracy).mean()))
