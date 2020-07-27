##################################################################################################
#Draft Code
#Author：CE JU
##################################################################################################
import numpy as np
import random
from random import shuffle

import h5py
import os


import torch
from torch import FloatTensor
from torch.autograd import Variable

from torch.nn.modules.loss import MSELoss
import torch.nn.functional as F
import torch.nn as nn


import warnings
warnings.filterwarnings('ignore')

#%% EEGNet Baseline 2
class eegNet(nn.Module):
    '''
    EEGNet architecture:
        The expected input during runtime is in a formate:
            miniBatch x 1 x electroldes x time points
        During initialization set:
            nChan : number of electrodes
            nTime : number of samples
            nClasses: number of classes
    '''                 
    def initialBlocks(self, dropoutP, *args, **kwargs):
        block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding = (0, self.C1 // 2 ), bias =False),
                nn.BatchNorm2d(self.F1),
                Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                     padding = 0, bias = False, max_norm = 1,
                                     groups=self.F1),
                nn.BatchNorm2d(self.F1 * self.D),
                nn.ELU(),
                nn.AvgPool2d((1,4), stride = 4),
                nn.Dropout(p = dropoutP))
        block2 = nn.Sequential(
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 22),
                                     padding = (0, 22//2) , bias = False,
                                     groups=self.F1* self.D),
                nn.Conv2d(self.F1 * self.D, self.F2, (1,1),
                          stride =1, bias = False, padding = 0),
                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1,8), stride = 8),
                nn.Dropout(p = dropoutP)
                )
        return nn.Sequential(block1, block2)

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                nn.Conv2d(inF, outF, kernalSize, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass = 2,
                 dropoutP = 0.25, F1=4, D = 2,
                 C1 = 100, *args, **kwargs):
        super(eegNet, self).__init__()
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.nTime = nTime
        self.nClass = nClass
        self.nChan = nChan
        self.C1 = C1

        self.firstBlocks = self.initialBlocks(dropoutP)
        self.fSize = self.calculateOutSize(self.firstBlocks, nChan, nTime)
        self.lastLayer = self.lastBlock(self.F2, nClass, (1, self.fSize[1]))

    def forward(self, x):
        x = self.firstBlocks(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)
		

def EEGNet_experients(cov_data_good, labels_good, cov_data, labels, index, fold, subject):

	cov_data_train = np.concatenate((cov_data_good, cov_data[index != fold]), axis=0)
	cov_data_train = cov_data_train.reshape((cov_data_train.shape[0], 1, cov_data_train.shape[1], cov_data_train.shape[2]))
	cov_data_test  = cov_data[index == fold].reshape((cov_data[index == fold].shape[0], 1, cov_data[index == fold].shape[1], cov_data[index == fold].shape[2]))


	input_data_train = Variable(torch.from_numpy(cov_data_train)).float()
	input_data_test = Variable(torch.from_numpy(cov_data_test)).float()
	target_train = Variable(torch.LongTensor(np.concatenate((labels_good,labels[index != fold]), axis = 0)))
	target_test  = Variable(torch.LongTensor(labels[index == fold]))

	model = eegNet(nChan=32, nTime=161)

	lr = 0.1
	old_loss = 1000
	iteration = 0

	while np.abs(old_loss) > 0.1:
		iteration += 1

		logits = model(input_data_train)
		output = F.log_softmax(logits, dim = -1)
		loss = F.nll_loss(output, target_train)
		pred = output.data.max(1, keepdim=True)[1]
		correct = pred.eq(target_train.data.view_as(pred)).long().cpu().sum()

		model.zero_grad()
		loss.backward()
		if iteration % 10 == 0:
			print('Iteration {} loss: {:4f}'.format(iteration, loss.item()))
			lr = max(0.99*lr, 0.005)

		with torch.no_grad():
			for param in model.parameters():
				param -= lr * param.grad

		if np.abs(loss.item()-old_loss) < 1e-6: break
		old_loss = loss.item()

	logits = model(input_data_test)
	output = F.log_softmax(logits, dim = -1)
	loss = F.nll_loss(output, target_test)
	pred = output.data.max(1, keepdim=True)[1]
	correct_test = pred.eq(target_test.data.view_as(pred)).long().cpu().sum()

	print('Subject {} (Fold {}) Classification Accuracy: {:4f}.'.format(subject, fold, correct_test.item()/pred.shape[0]))
	print('--------------------------------------')

	return correct_test.item()/pred.shape[0]



if __name__ == '__main__':

	np.random.seed(0)

	data=np.load('raw_data/normalized_original_epoch_data_train.npy')
	label=np.load('raw_data/train_label.npy')
	index = np.load('index.npy')

	good_subject_index =[0, 1, 6, 7, 14, 28, 30, 32, 33, 34, 41, 47, 51, 53, 54, 55, 59, 61, 69, 70, 71, 72, 79, 84, 85, 92, 103]
	cov_data_good = np.concatenate(data[good_subject_index], axis=0)
	labels_good = np.concatenate(label[good_subject_index], axis=0)

	Total_average_acc = []

	#bad_subject_index = [2, 8, 16, 17, 22, 23, 27, 35, 37, 38, 39, 40, 44, 46, 57, 62, 63, 66, 73, 75, 76, 77, 89, 95, 96, 98, 100, 101]
	#for bad_subject_index in [2, 8, 16, 17, 22, 23, 27, 35, 37, 38, 39, 40, 44, 46, 57, 62, 63, 66, 73, 75, 76, 77, 89, 95, 96, 98, 100, 101]:
	for bad_subject_index in [16]:

		cov_data_subject = data[bad_subject_index]
		labels_subject   = label[bad_subject_index]

		accuracy = []

		for fold in range(1, 6):
			for _ in range(1):
				accuracy.append(EEGNet_experients(cov_data_good, labels_good, cov_data_subject, labels_subject, index[bad_subject_index], fold, bad_subject_index))

		print('--------------------------------')
		print('Subject {}: Average Classification Accuracy: {:4f}.'.format(bad_subject_index, np.array(accuracy).mean()))
		Total_average_acc.append(np.array(accuracy).mean())

	print('-------------------------------------------------------------------')
	print('Total_Average_ACC: {:4f}'.format(np.mean(Total_average_acc)))


