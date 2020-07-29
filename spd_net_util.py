import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.module import Module


class SVD_opt(Function):

    def forward(self, input):
        Us = torch.zeros_like(input)
        Ss = torch.zeros((input.shape[0], input.shape[1])).double()
        for i in range(input.shape[0]):
            U, S, V = torch.svd(input[i, :, :])
            Ss[i, :] = S
            Us[i, :, :] = U

        self.Us = Us
        self.Ss = Ss
        # self.save_for_backward(input)
        return Us, Ss

    def backward(self, dLdV, dLdS):

        Ut = torch.transpose(self.Us, 1, 2)
        Ks = torch.zeros_like(dLdV)
        diag_dLdS = torch.zeros_like(dLdV)

        for i in range(dLdV.shape[0]):
            diagS = self.Ss[i, :]
            diagS = diagS.contiguous()
            vs_1 = diagS.view([diagS.shape[0], 1])
            vs_2 = diagS.view([1, diagS.shape[0]])
            K = 1.0 / (vs_1 - vs_2)
            # K.masked_fill(mask_diag, 0.0)
            K[K >= float("Inf")] = 0.0
            Ks[i, :, :] = K

            # diag_dLdS[i, :, :] = torch.diag(torch.diag(dLdS[i, :, :]))
            diag_dLdS[i, :, :] = torch.diag(dLdS[i, :])

        tmp = torch.transpose(Ks, 1, 2) * torch.matmul(Ut, dLdV)
        tmp = 0.5 * (tmp + torch.transpose(tmp, 1, 2)) + diag_dLdS
        grad = torch.matmul(self.Us, torch.matmul(tmp, Ut))  # checked

        return grad


class RecFunction_v2(Function):

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
            # res = torch.matmul(U, torch.matmul(torch.diag(max_S), torch.transpose(U, 0, 1)))
            # result[i, :, :] = res
            Ss[i, :]=S
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
            # K.masked_fill(mask_diag, 0.0)
            K[K >= float("Inf")] = 0.0
            Ks[i, :, :] = K

            diag_dLdS[i, :, :] = torch.diag(torch.diag(dLdS[i, :, :]))

        tmp = torch.transpose(Ks, 1, 2) * torch.matmul(Ut, dLdV)
        tmp = 0.5 * (tmp + torch.transpose(tmp, 1, 2)) + diag_dLdS
        grad = torch.matmul(self.Us, torch.matmul(tmp, Ut))  # checked

        return grad
class LogFunction_v2(Function):
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
            invSs[i, :, :] = torch.diag(1.0/S)

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
        dLdC = 0.5 * (dLdC + torch.transpose(dLdC, 1, 2)) # checked

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

        # print('log_mat_v2 backward')
        return grad


class LogFunction_v0(Function):
    def forward(self, input):
        Us = torch.zeros_like(input)
        Ss = torch.zeros((input.shape[0], input.shape[1])).double()
        mSs = torch.zeros_like(input)
        for i in range(input.shape[0]):
            U, S, V = torch.svd(input[i, :, :])

            Ss[i, :] = S
            Us[i, :, :] = U
            mSs[i, :, :] = torch.diag(torch.log(S))

        result = torch.matmul(Us, torch.matmul(mSs, torch.transpose(Us, 1, 2)))
        self.Us = Us
        self.Ss = Ss
        self.save_for_backward(input)
        return result

    def backward(self, grad_output):
        grad_output = grad_output.double()
        grad = torch.zeros_like(grad_output)
        d = grad_output.shape[1]
        # mask_diag = torch.ByteTensor(torch.eye(d).byte())
        for i in range(grad_output.shape[0]):
            dLdC = grad_output[i, :, :]
            dLdC = 0.5 * (dLdC + torch.transpose(dLdC, 0, 1))  # checked

            diagS = self.Ss[i, :]
            diagS = diagS.contiguous()
            U = self.Us[i, :, :]
            Ut = torch.transpose(U, 0, 1)

            diagLogS = torch.diag(torch.log(diagS))  # matrix
            diagInvS = torch.diag(1.0 / diagS)  # matrix

            dLdV = 2 * torch.matmul(dLdC, torch.matmul(U, diagLogS))  # [d, ind]
            dLdS_1 = torch.matmul(torch.matmul(Ut, dLdC), U)  # [ind, ind]
            dLdS = torch.matmul(diagInvS, dLdS_1)

            vs_1 = diagS.view([diagS.shape[0], 1])
            vs_2 = diagS.view([1, diagS.shape[0]])
            K = 1.0 / (vs_1 - vs_2)
            # K.masked_fill(mask_diag, 0.0)
            K[K >= float("Inf")] = 0.0


            tmp = torch.transpose(K, 0, 1) * torch.matmul(Ut, dLdV)
            tmp = 0.5 * (tmp + torch.transpose(tmp, 0, 1)) + torch.diag(torch.diag(dLdS))
            dzdx = torch.matmul(U, torch.matmul(tmp, Ut))  # checked
            grad[i, :, :] = dzdx
        # print('log_mat_v2 backward')
        return grad


class LogFunction(Function):

    def forward(ctx, input):
        numpy_input = input.numpy()
        if numpy_input.dtype != np.float64:
            numpy_input = numpy_input.astype(np.float64)

        u, s, v = np.linalg.svd(numpy_input)
        diag = np.zeros(numpy_input.shape, dtype=np.float64)
        n = numpy_input.shape[0]
        for i in range(n):
            diag[i, :, :] = np.diag(np.log(s[i, :]))

        result = np.matmul(u, np.matmul(diag, np.transpose(u, axes=[0, 2, 1])))  # checked
        ctx.save_for_backward(input)

        return torch.DoubleTensor(result)

    def backward(ctx, grad_output):
        np_grad_output = grad_output.numpy()
        numpy_input, = ctx.saved_tensors
        numpy_input = numpy_input.numpy()

        if numpy_input.dtype != np.float64:
            numpy_input = numpy_input.astype(np.float64)
        if np_grad_output.dtype != np.float64:
            np_grad_output = np_grad_output.astype(np.float64)

        grad = np.zeros(np_grad_output.shape, dtype=np.float64)
        u, s, v = np.linalg.svd(numpy_input)
        for i in range(np_grad_output.shape[0]):
            dLdC = np_grad_output[i, :, :]
            dLdC = dLdC.astype(dtype=np.float64)
            dLdC = 0.5 * (dLdC + np.transpose(dLdC))  # checked

            diagS = s[i, :]
            U = u[i, :, :]  #

            # thr = dLdC.shape[-1] * np.spacing(np.max(diagS))
            # ind = np.greater(diagS, thr)
            # diagS = diagS[ind]
            # U = U[:, ind]  # checked [d, ind]

            diagLogS = np.diag(np.log(diagS))  # matrix
            diagInvS = np.diag(1.0 / diagS)  # matrix

            dLdV = 2 * np.matmul(dLdC, np.matmul(U, diagLogS))  # [d, ind]
            dLdS_1 = np.matmul(np.matmul(np.transpose(U), dLdC), U)  # [ind, ind]
            dLdS = np.matmul(diagInvS, dLdS_1)

            # if np.sum(ind) == 1:
            #     # K = 1. / (S(1) * ones(1, Dmin) - (S(1) * ones(1, Dmin))');
            #     # K(eye(size(K, 1)) > 0)=0;
            #     print('rank is one!')

            # K is [ind, ind]
            K = 1.0 / (np.expand_dims(diagS, axis=-1) - np.expand_dims(diagS, axis=0))
            np.fill_diagonal(K, 0.0)
            K[np.isinf(K)] = 0.0

            tmp = np.transpose(K) * np.matmul(np.transpose(U), dLdV)
            tmp = 0.5 * (tmp + np.transpose(tmp)) + np.diag(np.diag(dLdS))
            dzdx = np.matmul(U, np.matmul(tmp, np.transpose(U)))
            grad[i, :, :] = dzdx  # checked

        return torch.DoubleTensor(grad)


class RecFunction_v0(Function):

    def forward(self, input):
        Us = torch.zeros_like(input)
        Ss = torch.zeros((input.shape[0], input.shape[1])).double()
        max_Ss = torch.zeros_like(input)
        max_Ids = torch.zeros_like(input)
        for i in range(input.shape[0]):
            U, S, V = torch.svd(input[i, :, :])
            eps = 0.0001
            max_S = torch.clamp(S, min=eps)
            max_Id = torch.ge(S, eps)
            # res = torch.matmul(U, torch.matmul(torch.diag(max_S), torch.transpose(U, 0, 1)))
            # result[i, :, :] = res
            Ss[i, :]=S
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
        grad = torch.zeros_like(grad_output)
        d = grad_output[1]
        # mask_diag = torch.ByteTensor(torch.eye(d).byte())
        for i in range(grad_output.shape[0]):
            dLdC = grad_output[i, :, :]
            dLdC = 0.5 * (dLdC + torch.transpose(dLdC, 0, 1))  # checked

            rec_s = self.max_Ss[i, :, :]
            rec_id = self.max_Ids[i, :, :]
            U = self.Us[i, :, :]
            Ut = torch.transpose(U, 0, 1)
            diagS = self.Ss[i, :]
            diagS = diagS.contiguous()

            dLdV = 2 * torch.matmul(torch.matmul(dLdC, U), rec_s)

            dLdS_1 = torch.matmul(torch.matmul(Ut, dLdC), U)
            dLdS = torch.matmul(rec_id, dLdS_1)  # checked

            vs_1 = diagS.view([diagS.shape[0], 1])
            vs_2 = diagS.view([1, diagS.shape[0]])
            K = 1.0 / (vs_1 - vs_2)
            # K.masked_fill(mask_diag, 0.0)
            K[K >= float("Inf")] = 0.0

            tmp = torch.transpose(K, 0, 1) * torch.matmul(Ut, dLdV)
            tmp = 0.5 * (tmp + torch.transpose(tmp, 0, 1)) + torch.diag(torch.diag(dLdS))
            dzdx = torch.matmul(U, torch.matmul(tmp, Ut))  # checked
            grad[i, :, :] = dzdx
        # print('rec_mat_v2 backward')
        return grad


def SVD_customed(input):
    return SVD_opt()(input)


def rec_mat_v2(input):
    return RecFunction_v2()(input)


def log_mat_v2(input):
    return LogFunction_v2()(input)


class RecFunction(Function):

    # @staticmethod
    def forward(ctx, input):
        numpy_input = input.numpy()
        if numpy_input.dtype != np.float64:
            numpy_input = numpy_input.astype(np.float64)
        u, s, v = np.linalg.svd(numpy_input)
        eps = 0.0001
        max_s = np.maximum(s, eps)
        n = numpy_input.shape[0]
        diag = np.zeros(numpy_input.shape, dtype=np.float64)
        # max_s = s
        for i in range(n):
            diag[i, :, :] = np.diag(max_s[i, :])

        result = np.matmul(u, np.matmul(diag, np.transpose(u, axes=[0, 2, 1])))  # checked
        ctx.save_for_backward(input)
        return torch.DoubleTensor(result)

    # @staticmethod
    def backward(ctx, grad_output):
        np_grad_output = grad_output.numpy()
        numpy_input, = ctx.saved_tensors
        numpy_input = numpy_input.numpy()

        if numpy_input.dtype != np.float64:
            numpy_input = numpy_input.astype(np.float64)
        if np_grad_output.dtype != np.float64:
            np_grad_output = np_grad_output.astype(np.float64)


        u, s, v = np.linalg.svd(numpy_input)
        eps = 0.0001
        max_s = np.maximum(s, eps)
        max_id = np.greater(s, eps).astype(dtype=np.float32)
        grad = np.zeros(np_grad_output.shape)
        for i in range(np_grad_output.shape[0]):
            dLdC = np_grad_output[i, :, :]
            dLdC = 0.5 * (dLdC + np.transpose(dLdC))  # checked

            rec_s = np.diag(max_s[i, :])  # checked
            rec_id = np.diag(max_id[i, :])
            U = u[i, :, :]
            dLdV = 2 * np.matmul(np.matmul(dLdC, U), rec_s)

            dLdS_1 = np.matmul(np.matmul(np.transpose(U), dLdC), U)
            dLdS = np.matmul(rec_id, dLdS_1)  # checked

            K = 1.0 / (np.expand_dims(s[i, :], axis=-1) - np.expand_dims(s[i, :], axis=0))
            np.fill_diagonal(K, 0.0)
            K[np.isinf(K)] = 0.0  # checked

            tmp = np.transpose(K) * np.matmul(np.transpose(U), dLdV)
            tmp = 0.5 * (tmp + np.transpose(tmp)) + np.diag(np.diag(dLdS))
            dzdx = np.matmul(U, np.matmul(tmp, np.transpose(U)))  # checked
            grad[i, :, :] = dzdx

        return torch.DoubleTensor(grad)
def rec_mat(input):
    return RecFunction()(input)
def log_mat(input):
    return LogFunction()(input)
def cal_riemann_grad_torch(X, U):
    '''

    :param X: the parameter
    :param U: the eculidean gradient
    :return: the riemann gradient
    '''
    #-- Matlab code
    # XtU = X'*U;
    # symXtU = 0.5 * (XtU + XtU');
    # Up = U - X * symXtU;

    #-- numpy code
    # XtU = np.matmul(np.transpose(X), U)
    # symXtU = 0.5*(XtU + np.transpose(XtU))
    # Up = U - np.matmul(X, symXtU)

    XtU = torch.matmul(torch.transpose(X, 0, 1), U)
    symXtU = 0.5*(XtU + torch.transpose(XtU, 0, 1))
    Up = U - torch.matmul(X, symXtU)
    return Up


def cal_retraction_torch(X, rU, t):
    """

    :param X: the parameter
    :param rU: the riemann gradient
    :param t: the learning rate
    :return: the retraction:
    """
    # matlab code
    # Y = X + t * U;
    # [Q, R] = qr(Y, 0);
    # Y = Q * diag(sign(diag(R)));

    # python code
    # Y = X - t*rU
    # Q, R = np.linalg.qr(Y, mode='reduced')
    # sR = np.diag(np.sign(np.diag(R)))
    # Y = np.matmul(Q, sR)

    Y = X - t*rU

    return Y


def update_para_riemann(X, U, t):
    Up = cal_riemann_grad(X, U)
    new_X = cal_retraction(X, Up, t)
    return new_X


def cal_riemann_grad(X, U):
    '''

    :param X: the parameter
    :param U: the eculidean gradient
    :return: the riemann gradient
    '''
    # XtU = X'*U;
    # symXtU = 0.5 * (XtU + XtU');
    # Up = U - X * symXtU;
    XtU = np.matmul(np.transpose(X), U)
    symXtU = 0.5*(XtU + np.transpose(XtU))
    Up = U - np.matmul(X, symXtU)

    return Up


def cal_retraction(X, rU, t):
    """

    :param X: the parameter
    :param rU: the riemann gradient
    :param t: the learning rate
    :return: the retraction:
    """
    # Y = X + t * U;
    # [Q, R] = qr(Y, 0);
    # Y = Q * diag(sign(diag(R)));
    Y = X - t*rU
    Q, R = np.linalg.qr(Y, mode='reduced')
    sR = np.diag(np.sign(np.diag(R)))
    Y = np.matmul(Q, sR)

    return Y