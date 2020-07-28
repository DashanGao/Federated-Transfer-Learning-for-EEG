import torch
from torch.autograd import Variable
import spd_net_util as util
import torch.nn.functional as F


class SPDNetwork(torch.nn.Module):

    def __init__(self):
        super(SPDNetwork, self).__init__()
        self.w_1_p = Variable(torch.randn(32, 4).double(), requires_grad=True)
        self.w_2_p = Variable(torch.randn(4, 4).double(), requires_grad=True)
        self.w_3_p = Variable(torch.randn(4, 4).double(), requires_grad=True)
        self.fc_w = Variable(torch.randn(16, 2).double(), requires_grad=True)

    def forward(self, input):
        batch_size = input.shape[0]
        w_1_pc = self.w_1_p.contiguous()
        w_1 = w_1_pc.view([1, w_1_pc.shape[0], w_1_pc.shape[1]])

        w_2_pc = self.w_2_p.contiguous()
        w_2 = w_2_pc.view([1, w_2_pc.shape[0], w_2_pc.shape[1]])

        w_3_pc = self.w_3_p.contiguous()
        w_3 = w_3_pc.view([1, w_3_pc.shape[0], w_3_pc.shape[1]])

        w_tX = torch.matmul(torch.transpose(w_1, dim0=1, dim1=2), input)
        w_tXw = torch.matmul(w_tX, w_1)
        X_1 = util.rec_mat_v2(w_tXw)

        w_tX = torch.matmul(torch.transpose(w_2, dim0=1, dim1=2), X_1)
        w_tXw = torch.matmul(w_tX, w_2)
        X_2 = util.rec_mat_v2(w_tXw)

        w_tX = torch.matmul(torch.transpose(w_3, dim0=1, dim1=2), X_2)
        w_tXw = torch.matmul(w_tX, w_3)
        X_3 = util.log_mat_v2(w_tXw)

        feat = X_3.view([batch_size, -1])  # [batch_size, d]
        logits = torch.matmul(feat, self.fc_w)  # [batch_size, num_class]

        return logits

    def update_para(self, lr):
        egrad_w1 = self.w_1_p.grad.data.numpy()
        egrad_w2 = self.w_2_p.grad.data.numpy()
        egrad_w3 = self.w_3_p.grad.data.numpy()
        w_1_np = self.w_1_p.data.numpy()
        w_2_np = self.w_2_p.data.numpy()
        w_3_np = self.w_3_p.data.numpy()

        new_w_3 = util.update_para_riemann(w_3_np, egrad_w3, lr)
        new_w_2 = util.update_para_riemann(w_2_np, egrad_w2, lr)
        new_w_1 = util.update_para_riemann(w_1_np, egrad_w1, lr)

        self.w_1_p.data.copy_(torch.DoubleTensor(new_w_1))
        self.w_2_p.data.copy_(torch.DoubleTensor(new_w_2))
        self.w_3_p.data.copy_(torch.DoubleTensor(new_w_3))

        self.fc_w.data -= lr * self.fc_w.grad.data
        # Manually zero the gradients after updating weights
        self.w_1_p.grad.data.zero_()
        self.w_2_p.grad.data.zero_()
        self.w_3_p.grad.data.zero_()
        self.fc_w.grad.data.zero_()
        # print('finished')


class SPDNetwork_1(torch.nn.Module):

    def __init__(self):
        super(SPDNetwork_1, self).__init__()

        self.w_1_p = Variable(torch.randn(32, 32).double(), requires_grad=True)
        self.w_2_p = Variable(torch.randn(32, 16).double(), requires_grad=True)
        self.w_3_p = Variable(torch.randn(16, 4).double(), requires_grad=True)
        self.fc_w = Variable(torch.randn(16, 2).double(), requires_grad=True)

        # self.w_1_p = Variable(nn.init.zeros_(torch.empty(32,16)).double(), requires_grad=True)
        # self.w_2_p = Variable(nn.init.zeros_(torch.empty(16,8)).double(), requires_grad=True)
        # self.w_3_p = Variable(nn.init.eye_(torch.empty(8,4)).double(), requires_grad=True)
        # self.fc_w = Variable(nn.init.zeros_(torch.empty(16,2)).double(), requires_grad=True)

    def forward(self, input):
        batch_size = input.shape[0]
        w_1_pc = self.w_1_p.contiguous()
        w_1 = w_1_pc.view([1, w_1_pc.shape[0], w_1_pc.shape[1]])

        w_2_pc = self.w_2_p.contiguous()
        w_2 = w_2_pc.view([1, w_2_pc.shape[0], w_2_pc.shape[1]])

        w_3_pc = self.w_3_p.contiguous()
        w_3 = w_3_pc.view([1, w_3_pc.shape[0], w_3_pc.shape[1]])

        w_tX = torch.matmul(torch.transpose(w_1, dim0=1, dim1=2), input)
        w_tXw = torch.matmul(w_tX, w_1)
        X_1 = util.rec_mat_v2(w_tXw)
        # X_1 = w_tXw

        w_tX = torch.matmul(torch.transpose(w_2, dim0=1, dim1=2), X_1)
        w_tXw = torch.matmul(w_tX, w_2)
        X_2 = util.rec_mat_v2(w_tXw)
        # X_2 = w_tXw

        w_tX = torch.matmul(torch.transpose(w_3, dim0=1, dim1=2), X_2)
        w_tXw = torch.matmul(w_tX, w_3)
        X_3 = util.log_mat_v2(w_tXw)

        feat = X_3.view([batch_size, -1])  # [batch_size, d]
        logits = torch.matmul(feat, self.fc_w)  # [batch_size, num_class]
        output = F.log_softmax(logits, dim=-1)
        return output, feat

    def update_para(self, lr):
        egrad_w1 = self.w_1_p.grad.data.numpy()
        egrad_w2 = self.w_2_p.grad.data.numpy()
        egrad_w3 = self.w_3_p.grad.data.numpy()
        w_1_np = self.w_1_p.data.numpy()
        w_2_np = self.w_2_p.data.numpy()
        w_3_np = self.w_3_p.data.numpy()

        new_w_3 = util.update_para_riemann(w_3_np, egrad_w3, lr)
        new_w_2 = util.update_para_riemann(w_2_np, egrad_w2, lr)
        new_w_1 = util.update_para_riemann(w_1_np, egrad_w1, lr)

        self.w_1_p.data.copy_(torch.DoubleTensor(new_w_1))
        self.w_2_p.data.copy_(torch.DoubleTensor(new_w_2))
        self.w_3_p.data.copy_(torch.DoubleTensor(new_w_3))

        # self.fc_w.data -= lr * self.fc_w.grad.data
        # Manually zero the gradients after updating weights
        self.w_1_p.grad.data.zero_()
        self.w_2_p.grad.data.zero_()
        self.w_3_p.grad.data.zero_()

        return self.fc_w.grad.data

    def second_update_para(self, lr, average_grad):
        self.fc_w.data -= lr * average_grad
        self.fc_w.grad.data.zero_()


class SPDNetwork_2(torch.nn.Module):

    def __init__(self):
        super(SPDNetwork_2, self).__init__()

        self.w_1_p = Variable(torch.randn(32, 4).double(), requires_grad=True)
        self.w_2_p = Variable(torch.randn(4, 4).double(), requires_grad=True)
        self.w_3_p = Variable(torch.randn(4, 4).double(), requires_grad=True)
        self.fc_w = Variable(torch.randn(16, 2).double(), requires_grad=True)

    def forward(self, input):
        batch_size = input.shape[0]
        w_1_pc = self.w_1_p.contiguous()
        w_1 = w_1_pc.view([1, w_1_pc.shape[0], w_1_pc.shape[1]])

        w_2_pc = self.w_2_p.contiguous()
        w_2 = w_2_pc.view([1, w_2_pc.shape[0], w_2_pc.shape[1]])

        w_3_pc = self.w_3_p.contiguous()
        w_3 = w_3_pc.view([1, w_3_pc.shape[0], w_3_pc.shape[1]])

        w_tX = torch.matmul(torch.transpose(w_1, dim0=1, dim1=2), input)
        w_tXw = torch.matmul(w_tX, w_1)
        X_1 = util.rec_mat_v2(w_tXw)
        # X_1 = w_tXw

        w_tX = torch.matmul(torch.transpose(w_2, dim0=1, dim1=2), X_1)
        w_tXw = torch.matmul(w_tX, w_2)
        X_2 = util.rec_mat_v2(w_tXw)
        # X_2 = w_tXw

        w_tX = torch.matmul(torch.transpose(w_3, dim0=1, dim1=2), X_2)
        w_tXw = torch.matmul(w_tX, w_3)
        X_3 = util.log_mat_v2(w_tXw)

        feat = X_3.view([batch_size, -1])  # [batch_size, d]
        logits = torch.matmul(feat, self.fc_w)  # [batch_size, num_class]
        output = F.log_softmax(logits, dim=-1)
        return output, feat

    def update_para(self, lr):
        egrad_w1 = self.w_1_p.grad.data.numpy()
        egrad_w2 = self.w_2_p.grad.data.numpy()
        egrad_w3 = self.w_3_p.grad.data.numpy()
        w_1_np = self.w_1_p.data.numpy()
        w_2_np = self.w_2_p.data.numpy()
        w_3_np = self.w_3_p.data.numpy()

        new_w_3 = util.update_para_riemann(w_3_np, egrad_w3, lr)
        new_w_2 = util.update_para_riemann(w_2_np, egrad_w2, lr)
        new_w_1 = util.update_para_riemann(w_1_np, egrad_w1, lr)

        self.w_1_p.data.copy_(torch.DoubleTensor(new_w_1))
        self.w_2_p.data.copy_(torch.DoubleTensor(new_w_2))
        self.w_3_p.data.copy_(torch.DoubleTensor(new_w_3))

        # self.fc_w.data -= lr * self.fc_w.grad.data
        # Manually zero the gradients after updating weights
        self.w_1_p.grad.data.zero_()
        self.w_2_p.grad.data.zero_()
        self.w_3_p.grad.data.zero_()

        return self.fc_w.grad.data

    def second_update_para(self, lr, average_grad):
        self.fc_w.data -= lr * average_grad
        self.fc_w.grad.data.zero_()


class SPDNetwork_3(torch.nn.Module):

    def __init__(self):
        super(SPDNetwork_3, self).__init__()

        self.w_1_p = Variable(torch.randn(32, 4).double(), requires_grad=True)
        self.w_3_p = Variable(torch.randn(4, 4).double(), requires_grad=True)
        self.fc_w = Variable(torch.randn(16, 2).double(), requires_grad=True)

        # self.w_1_p = Variable(nn.init.zeros_(torch.empty(32,16)).double(), requires_grad=True)
        # self.w_2_p = Variable(nn.init.zeros_(torch.empty(16,8)).double(), requires_grad=True)
        # self.w_3_p = Variable(nn.init.eye_(torch.empty(8,4)).double(), requires_grad=True)
        # self.fc_w = Variable(nn.init.zeros_(torch.empty(16,2)).double(), requires_grad=True)

    def forward(self, input):
        batch_size = input.shape[0]
        w_1_pc = self.w_1_p.contiguous()
        w_1 = w_1_pc.view([1, w_1_pc.shape[0], w_1_pc.shape[1]])

        w_3_pc = self.w_3_p.contiguous()
        w_3 = w_3_pc.view([1, w_3_pc.shape[0], w_3_pc.shape[1]])

        w_tX = torch.matmul(torch.transpose(w_1, dim0=1, dim1=2), input)
        w_tXw = torch.matmul(w_tX, w_1)
        X_1 = util.rec_mat_v2(w_tXw)
        # X_1 = w_tXw

        w_tX = torch.matmul(torch.transpose(w_3, dim0=1, dim1=2), X_1)
        w_tXw = torch.matmul(w_tX, w_3)
        X_3 = util.log_mat_v2(w_tXw)

        feat = X_3.view([batch_size, -1])  # [batch_size, d]
        logits = torch.matmul(feat, self.fc_w)  # [batch_size, num_class]

        return logits, feat

    def update_para(self, lr):
        egrad_w1 = self.w_1_p.grad.data.numpy()
        egrad_w3 = self.w_3_p.grad.data.numpy()
        w_1_np = self.w_1_p.data.numpy()
        w_3_np = self.w_3_p.data.numpy()

        new_w_3 = util.update_para_riemann(w_3_np, egrad_w3, lr)
        new_w_1 = util.update_para_riemann(w_1_np, egrad_w1, lr)

        self.w_1_p.data.copy_(torch.DoubleTensor(new_w_1))
        self.w_3_p.data.copy_(torch.DoubleTensor(new_w_3))

        # self.fc_w.data -= lr * self.fc_w.grad.data
        # Manually zero the gradients after updating weights
        self.w_1_p.grad.data.zero_()
        self.w_3_p.grad.data.zero_()

        return self.fc_w.grad.data

    def second_update_para(self, lr, average_grad):
        self.fc_w.data -= lr * average_grad
        self.fc_w.grad.data.zero_()
