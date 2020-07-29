##################################################################################################
# SPDNet model definition
# Author：Ce Ju, Dashan Gao
# Date  : July 29, 2020
# Paper : Ce Ju et al., Federated Transfer Learning for EEG Signal Classification, IEEE EMBS 2020.
# Description: Source domain includes all good subjects, target domain is the bad subject.
##################################################################################################

import torch
from torch.autograd import Variable
import SPDNet_utils as util
import torch.nn.functional as F

torch.manual_seed(0)


class SPDNetwork_1(torch.nn.Module):
    """
    A sub-class of SPDNetwork with network structure of manifold reduction layers: [(32, 32), (32, 16), (16, 4)]
    """

    def __init__(self):
        super(SPDNetwork_1, self).__init__()

        self.w_1_p = Variable(torch.randn(32, 32).double(), requires_grad=True)
        self.w_2_p = Variable(torch.randn(32, 16).double(), requires_grad=True)
        self.w_3_p = Variable(torch.randn(16, 4).double(), requires_grad=True)
        self.fc_w = Variable(torch.randn(16, 2).double(), requires_grad=True)

    def forward(self, input):
        """
        Forward propagation
        :param input:
        :return:
                output: the predicted probability of the model.
                feat: feature in the common subspace for feature alignment.
        """
        batch_size = input.shape[0]

        output = input
        # Forward propagation of local model
        for idx, w in enumerate([self.w_1_p, self.w_2_p]):
            w = w.contiguous().view(1, w.shape[0], w.shape[1])
            w_tX = torch.matmul(torch.transpose(w, dim0=1, dim1=2), output)
            w_tXw = torch.matmul(w_tX, w)
            output = util.rec_mat(w_tXw)

        w_3 = self.w_3_p.contiguous().view([1, self.w_3_p.shape[0], self.w_3_p.shape[1]])
        w_tX = torch.matmul(torch.transpose(w_3, dim0=1, dim1=2), output)
        w_tXw = torch.matmul(w_tX, w_3)
        X_3 = util.log_mat(w_tXw)

        feat = X_3.view([batch_size, -1])  # [batch_size, d]
        logits = torch.matmul(feat, self.fc_w)  # [batch_size, num_class]
        output = F.log_softmax(logits, dim=-1)
        return output, feat

    def update_all_layers(self, lr):
        """
        Update all layers for local single party training.
        :param lr: learning rate
        :return: None
        """
        update_manifold_reduction_layer(lr, [self.w_1_p, self.w_2_p, self.w_3_p])
        self.fc_w.data -= lr * self.fc_w.grad.data
        self.fc_w.grad.data.zero_()

    def update_manifold_reduction_layer(self, lr):
        """
        Update the manifold reduction layers
        :param lr: learning rate
        :return: None
        """
        update_manifold_reduction_layer(lr, [self.w_1_p, self.w_2_p, self.w_3_p])

    def update_federated_layer(self, lr, average_grad):
        """
        Update the federated layer.
        :param lr: Learning rate
        :param average_grad: the average gradient of the federated layer of all participants
        :return: None
        """
        self.fc_w.data -= lr * average_grad
        self.fc_w.grad.data.zero_()


class SPDNetwork_2(torch.nn.Module):
    """
    A sub-class of SPDNetwork with network structure of manifold reduction layers: [(32, 4), (4, 4), (4, 4)]
    """
    def __init__(self):
        super(SPDNetwork_2, self).__init__()

        self.w_1_p = Variable(torch.randn(32, 4).double(), requires_grad=True)
        self.w_2_p = Variable(torch.randn(4, 4).double(), requires_grad=True)
        self.w_3_p = Variable(torch.randn(4, 4).double(), requires_grad=True)
        self.fc_w = Variable(torch.randn(16, 2).double(), requires_grad=True)

    def forward(self, input):
        """
        Forward propagation
        :param input:
        :return:
                output: the predicted probability of the model.
                feat: feature in the common subspace for feature alignment.
        """
        batch_size = input.shape[0]
        output = input
        # Forward propagation of local model
        for idx, w in enumerate([self.w_1_p, self.w_2_p]):
            w = w.contiguous().view(1, w.shape[0], w.shape[1])
            w_tX = torch.matmul(torch.transpose(w, dim0=1, dim1=2), output)
            w_tXw = torch.matmul(w_tX, w)
            output = util.rec_mat(w_tXw)

        w_3 = self.w_3_p.contiguous().view([1, self.w_3_p.shape[0], self.w_3_p.shape[1]])
        w_tX = torch.matmul(torch.transpose(w_3, dim0=1, dim1=2), output)
        w_tXw = torch.matmul(w_tX, w_3)
        X_3 = util.log_mat(w_tXw)

        feat = X_3.view([batch_size, -1])  # [batch_size, d]
        logits = torch.matmul(feat, self.fc_w)  # [batch_size, num_class]
        output = F.log_softmax(logits, dim=-1)
        return output, feat

    def update_all_layers(self, lr):
        """
        Update all layers for local single party training.
        :param lr: learning rate
        :return: None
        """
        update_manifold_reduction_layer(lr, [self.w_1_p, self.w_2_p, self.w_3_p])
        self.fc_w.data -= lr * self.fc_w.grad.data
        self.fc_w.grad.data.zero_()

    def update_manifold_reduction_layer(self, lr):
        """
        Update the manifold reduction layers
        :param lr: learning rate
        :return: None
        """
        update_manifold_reduction_layer(lr, [self.w_1_p, self.w_2_p, self.w_3_p])

    def update_federated_layer(self, lr, average_grad):
        """
        Update the federated layer.
        :param lr: Learning rate
        :param average_grad: the average gradient of the federated layer of all participants
        :return: None
        """
        self.fc_w.data -= lr * average_grad
        self.fc_w.grad.data.zero_()


# Define the SPDNetwork the same as SPDNetwork_2 for convenience.
SPDNetwork = SPDNetwork_2


def update_manifold_reduction_layer(lr, params_list):
    """
    Update parameters of the participant-specific parameters, here are [self.w_1_p, self.w_2_p, self.w_3_p]
    :param lr: learning rate
    :param params_list: parameter list
    :return: None
    """
    for w in params_list:
        grad_w_np = w.grad.data.numpy()
        w_np = w.data.numpy()
        updated_w = util.update_para_riemann(w_np, grad_w_np, lr)
        w.data.copy_(torch.DoubleTensor(updated_w))
        # Manually zero the gradients after updating weights
        w.grad.data.zero_()
