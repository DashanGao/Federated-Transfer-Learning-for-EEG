##################################################################################################
# This code originally comes from: https://github.com/YirongMao/SPDNet/blob/master/spd_net_util.py by YirongMao.
# We use rec_mat_v2(), log_mat_v2() and update_para_riemann() in this project.
# Author：Yirong Mao, Ce Ju, Dashan Gao
# Date  : July 29, 2020
# Paper : Ce Ju et al., Federated Transfer Learning for EEG Signal Classification, IEEE EMBS 2020.
# Description: Source domain includes all good subjects, target domain is the bad subject.
##################################################################################################

import torch
from torch.autograd import Function
import numpy as np


class RecFunction(Function):

    def forward(self, input):
        Us = torch.zeros_like(input)
        Ss = torch.zeros((input.shape[0], input.shape[1])).double()
        max_Ss = torch.zeros_like(input)
        max_Ids = torch.zeros_like(input)
        for i in range(input.shape[0]):
            U, S, V = torch.svd(input[i, :, :])
            eps = 0.0001
            max_S = torch.clamp(S, min=eps)
            max_Id = torch.ge(S, eps).float()
            Ss[i, :] = S
            Us[i, :, :] = U
            max_Ss[i, :, :] = torch.diag(max_S)
            max_Ids[i, :, :] = torch.diag(max_Id)

        result = torch.matmul(Us, torch.matmul(max_Ss, torch.transpose(Us, 1, 2)))
        self.Us = Us
        self.Ss = Ss
        self.max_Ss = max_Ss
        self.max_Ids = max_Ids
        self.save_for_backward(input)
        return result

    def backward(self, grad_output):
        Ks = torch.zeros_like(grad_output)

        dLdC = grad_output
        dLdC = 0.5 * (dLdC + torch.transpose(dLdC, 1, 2))  # checked
        Ut = torch.transpose(self.Us, 1, 2)
        dLdV = 2 * torch.matmul(torch.matmul(dLdC, self.Us), self.max_Ss)
        dLdS_1 = torch.matmul(torch.matmul(Ut, dLdC), self.Us)
        dLdS = torch.matmul(self.max_Ids, dLdS_1)  # checked

        diag_dLdS = torch.zeros_like(grad_output)
        for i in range(grad_output.shape[0]):
            diagS = self.Ss[i, :]
            diagS = diagS.contiguous()
            vs_1 = diagS.view([diagS.shape[0], 1])
            vs_2 = diagS.view([1, diagS.shape[0]])
            K = 1.0 / (vs_1 - vs_2)
            K[K >= float("Inf")] = 0.0
            Ks[i, :, :] = K
            diag_dLdS[i, :, :] = torch.diag(torch.diag(dLdS[i, :, :]))

        tmp = torch.transpose(Ks, 1, 2) * torch.matmul(Ut, dLdV)
        tmp = 0.5 * (tmp + torch.transpose(tmp, 1, 2)) + diag_dLdS
        grad = torch.matmul(self.Us, torch.matmul(tmp, Ut))  # checked

        return grad


class LogFunction(Function):
    def forward(self, input):
        Us = torch.zeros_like(input)
        Ss = torch.zeros((input.shape[0], input.shape[1])).double()
        logSs = torch.zeros_like(input)
        invSs = torch.zeros_like(input)
        for i in range(input.shape[0]):
            U, S, V = torch.svd(input[i, :, :])
            Ss[i, :] = S
            Us[i, :, :] = U
            logSs[i, :, :] = torch.diag(torch.log(S))
            invSs[i, :, :] = torch.diag(1.0 / S)
        result = torch.matmul(Us, torch.matmul(logSs, torch.transpose(Us, 1, 2)))
        self.Us = Us
        self.Ss = Ss
        self.logSs = logSs
        self.invSs = invSs
        self.save_for_backward(input)
        return result

    def backward(self, grad_output):
        grad_output = grad_output.double()
        Ks = torch.zeros_like(grad_output)
        dLdC = grad_output
        dLdC = 0.5 * (dLdC + torch.transpose(dLdC, 1, 2))  # checked
        Ut = torch.transpose(self.Us, 1, 2)
        dLdV = 2 * torch.matmul(dLdC, torch.matmul(self.Us, self.logSs))  # [d, ind]
        dLdS_1 = torch.matmul(torch.matmul(Ut, dLdC), self.Us)  # [ind, ind]
        dLdS = torch.matmul(self.invSs, dLdS_1)
        diag_dLdS = torch.zeros_like(grad_output)
        for i in range(grad_output.shape[0]):
            diagS = self.Ss[i, :]
            diagS = diagS.contiguous()
            vs_1 = diagS.view([diagS.shape[0], 1])
            vs_2 = diagS.view([1, diagS.shape[0]])
            K = 1.0 / (vs_1 - vs_2)
            # K.masked_fill(mask_diag, 0.0)
            K[K >= float("Inf")] = 0.0
            Ks[i, :, :] = K

            diag_dLdS[i, :, :] = torch.diag(torch.diag(dLdS[i, :, :]))

        tmp = torch.transpose(Ks, 1, 2) * torch.matmul(Ut, dLdV)
        tmp = 0.5 * (tmp + torch.transpose(tmp, 1, 2)) + diag_dLdS
        grad = torch.matmul(self.Us, torch.matmul(tmp, Ut))  # checked
        return grad


def rec_mat(input):
    return RecFunction()(input)


def log_mat(input):
    return LogFunction()(input)


def update_para_riemann(X, U, t):
    Up = cal_riemann_grad(X, U)
    new_X = cal_retraction(X, Up, t)
    return new_X


def cal_riemann_grad(X, U):
    """
    Calculate Riemann gradient.
    :param X: the parameter
    :param U: the eculidean gradient
    :return: the riemann gradient
    """
    # XtU = X'*U;
    XtU = np.matmul(np.transpose(X), U)
    # symXtU = 0.5 * (XtU + XtU');
    symXtU = 0.5 * (XtU + np.transpose(XtU))
    # Up = U - X * symXtU;
    Up = U - np.matmul(X, symXtU)
    return Up


def cal_retraction(X, rU, t):
    """
    Calculate the retraction value
    :param X: the parameter
    :param rU: the riemann gradient
    :param t: the learning rate
    :return: the retraction:
    """
    # Y = X + t * U;
    # [Q, R] = qr(Y, 0);
    # Y = Q * diag(sign(diag(R)));
    Y = X - t * rU
    Q, R = np.linalg.qr(Y, mode='reduced')
    sR = np.diag(np.sign(np.diag(R)))
    Y = np.matmul(Q, sR)

    return Y
