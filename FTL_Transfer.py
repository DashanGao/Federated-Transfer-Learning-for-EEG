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
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


def transfer_SPD(cov_data_1, cov_data_2, labels_1, labels_2):
    np.random.seed(0)

    # Shuffle data

    cov_data_1, labels_1 = shuffle_data(cov_data_1, labels_1)
    cov_data_2, labels_2 = shuffle_data(cov_data_2, labels_2)
    print(cov_data_1.shape, cov_data_2.shape)

    # Train test split
    train_data_1_num = cov_data_1.shape[0]
    cov_data_train_1 = cov_data_1[0:cov_data_1.shape[0], :, :]

    train_data_2_num = int(np.floor(cov_data_2.shape[0] * 0.8))
    cov_data_train_2 = cov_data_2[0:train_data_2_num, :, :]
    cov_data_test_2 = cov_data_2[train_data_2_num:cov_data_2.shape[0], :, :]

    print('split_num_for_test: ', train_data_2_num)
    print('rest_num_for_test: ', labels_2.shape[0] - train_data_2_num)
    print('-------------------------------------------------------')

    # Convert training data to torch variables.
    data_train_1 = Variable(torch.from_numpy(cov_data_train_1)).double()
    data_train_2 = Variable(torch.from_numpy(cov_data_train_2)).double()
    data_test_2 = Variable(torch.from_numpy(cov_data_test_2)).double()

    target_train_1 = Variable(torch.LongTensor(labels_1[0:train_data_1_num]))
    target_train_2 = Variable(torch.LongTensor(labels_2[0:train_data_2_num]))
    target_test_2 = Variable(torch.LongTensor(labels_2[train_data_2_num:labels_2.shape[0]]))

    # Initialize Model
    model_1 = model.SPDNetwork_1()
    model_2 = model.SPDNetwork_2()

    # Start training
    old_loss = 0
    lr, lr_1, lr_2 = 0.1, 0.1, 0.1
    train_accuracy_1, train_accuracy_2, test_accuracy_2 = 0, 0, 0
    for iteration in range(200):

        output_1, feat_1 = model_1(data_train_1)
        output_2, feat_2 = model_2(data_train_2)

        # Index of positive/negative labels
        feat_1_positive, feat_1_negative = split_class_feat(feat_1, target_train_1)
        feat_2_positive, feat_2_negative = split_class_feat(feat_2, target_train_2)

        # MMD knowledge transfer
        mmd = MMD_loss('rbf', kernel_mul=2.0)

        loss = F.nll_loss(output_1, target_train_1) + F.nll_loss(output_2, target_train_2) + \
               1 * mmd.forward(feat_1_positive, feat_2_positive) + \
               1 * mmd.forward(feat_1_negative, feat_2_negative)

        loss.backward()

        # Federated Averaging for model update
        grad_1 = model_1.update_para(lr_1)
        grad_2 = model_2.update_para(lr_2)
        average_grad = (grad_1 + grad_2) / 2

        model_1.second_update_para(lr, average_grad)
        model_2.second_update_para(lr, average_grad)

        # Evaluate model performance
        if iteration % 1 == 0:
            # Accuracy of two models
            pred_1 = output_1.data.max(1, keepdim=True)[1]
            pred_2 = output_2.data.max(1, keepdim=True)[1]
            train_accuracy_1 = pred_1.eq(target_train_1.data.view_as(pred_1)).long().cpu().sum().float() / pred_1.shape[
                0]
            train_accuracy_2 = pred_2.eq(target_train_2.data.view_as(pred_2)).long().cpu().sum().float() / pred_2.shape[
                0]
            print('Iteration {}: Trainning Accuracy for Source Task Model: {:.4f} / Target Task Model: {:.4f}'.format(
                iteration,
                train_accuracy_1,
                train_accuracy_2))

            logits_2, _ = model_2(data_test_2)
            output_2 = F.log_softmax(logits_2, dim=-1)
            loss_2 = F.nll_loss(output_2, target_test_2)
            pred_2 = output_2.data.max(1, keepdim=True)[1]
            test_accuracy_2 = pred_2.eq(target_test_2.data.view_as(pred_2)).long().cpu().sum().float() / pred_2.shape[0]
            print('Testing Accuracy for Model 2: {:.4f}'.format(test_accuracy_2))

        if np.abs(loss.item() - old_loss) < 1e-4:
            break
        old_loss = loss.item()

        # Update learning rates
        if iteration % 50 == 0:
            lr = max(0.98 * lr, 0.01)
            lr_1 = max(0.98 * lr_1, 0.01)
            lr_2 = max(0.98 * lr_2, 0.01)

    return test_accuracy_2


def load_data(data_file, label_file):
    """
    Load data training data
    :param data_file: training samples of all subjects
    :param label_file: labels of training samples of all subjects
    :return: data and labels of the good subjects as well as one specific bad subject.
    """
    data = np.load(data_file)
    label = np.load(label_file)
    GOOD_SUBJECT_IDS = [0, 1, 6, 7, 14, 28, 30, 32, 33, 34, 41, 47, 51, 53, 54, 55, 59, 61, 69, 70, 71, 72,
                        79, 84, 85, 92, 99, 103]
    BAD_SUBJECT_ID = 100

    # Good Subjects
    good_subj_data = np.concatenate(data[GOOD_SUBJECT_IDS], axis=0)
    good_subj_label = np.concatenate(label[GOOD_SUBJECT_IDS], axis=0)

    # Bad Subject
    bad_subj_data = data[BAD_SUBJECT_ID]
    bad_subj_label = label[BAD_SUBJECT_ID]

    return good_subj_data, good_subj_label, bad_subj_data, bad_subj_label


def split_class_feat(feat, target):
    """
    Split the features according to the true label of the training samples. This is meant to apply MMD of the
    features of each class.
    :param feat: features
    :param target: targets/ labels
    :return: features of positive calss and features of negative class
    """
    positive_index, negative_index = np.array(target) == 1, np.array(target) == 0
    positive_feat = feat[positive_index].detach().numpy()
    negative_feat = feat[negative_index].detach().numpy()
    # Convert to Variable for further training.
    positive_feat = Variable(torch.from_numpy(positive_feat)).double()
    negative_feat = Variable(torch.from_numpy(negative_feat)).double()
    return positive_feat, negative_feat


def shuffle_data(x, y):
    """
    Shuffle the data and labels.
    :param x: data
    :param y: targets
    :return: shuffled data adn labels
    """
    idx = np.random.permutation(x.shape[0])
    return x[idx, :, :], y[idx]


if __name__ == '__main__':

    np.random.seed(0)

    # Load data of good subjects and bad subjects.
    good_subj_data, good_subj_label, bad_subj_data, bad_subj_label = \
        load_data('raw_data/normalized_original_train_sample.npy', 'raw_data/train_label.npy')

    accuracy_recorder = []
    for _ in range(10):
        # Conduct federated transfer learning over good and bad subjects.
        accuracy = transfer_SPD(good_subj_data, bad_subj_data, good_subj_label, bad_subj_label)
        accuracy_recorder.append(accuracy)

    print('All Accuracy: ', accuracy_recorder)
    print('SPD Riemannian Average Classification Accuracy: {:4f}.'.format(np.array(accuracy_recorder).mean()))
