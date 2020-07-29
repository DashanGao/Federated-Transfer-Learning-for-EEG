# Federated Transfer Learning For EEG Signal Classification
Authors: Ce Ju, Dashan Gao, Ravikiran Mane, Ben Tan, Yang Liu and Cuntai Guan

---

## Introduction

<!--- ![Federated Learning](https://github.com/DashanGao/Federated-Transfer-Leraning-for-EEG/blob/master/imgs/federated_learning.png =250*250)![Federated Learning EEG](https://github.com/DashanGao/Federated-Transfer-Leraning-for-EEG/blob/master/imgs/federated_learning_eeg.png =250*250) --->

The success of deep learning (DL) methods in the Brain-Computer Interfaces (BCI) field for classification of 
electroencephalographic (EEG) recordings have been restricted by the lack of large datasets. Privacy concerns 
associated with the EEG signals limit the possibility to construct a large EEG- BCI datasets by the conglomeration of 
multiple small ones for jointly training the machine learning model. Hence, in this paper, we propose a novel 
privacy-preserving DL architecture named federated transfer learning (FTL) for EEG classification that is based on the 
federated learning framework. Working with the single-trial covariance matrix, the proposed architecture extracts common 
discriminative information from multi-subject EEG data with the help of domain adaption techniques. We evaluate 
the performance of the proposed architecture on the PhysioNet dataset for 2-class motor imagery classification. 
While avoiding the actual data sharing, our FTL approach achieves 2% higher classification accuracy in a 
subject-adaptive analysis. Also, in the absence of multi-subject data, our architecture provides 6% better accuracy 
compared to other state-of-the-art DL architectures.

---

## Network Architecture

Our proposed architecture consists of following 4 layers:
manifold reduction layer, common embedded space, tangent projection layer and federated layer. 
The purpose of each layer is as follows:

1. **Manifold reduction layer**: Spatial covariance matrices are always assumed to be on the high-dimensional 
SPD manifolds. This layer is the linear map from the high-dimensional SPD manifold to the low-dimensional one 
with undetermined weights for learning.

2. **Common embedded space**: The common space is the low-dimensional SPD manifold whose elements are reduced from 
each high-dimensional SPD manifolds, which is designed only for the transfer learning setting.

3. **Tangent projection layer**: This layer is to project the matrices on SPD manifolds to its tangent space, 
which is a local linear approximation of the curved space.

4. **Federated layer**: Deep neural networks are implemented in this layer. For the transfer learning setting, 
parameters of neural networks are updated by the federated aggregation.

---

## How To Run The Code

Please install pyRiemann and mne

pyRiemann: https://github.com/alexandrebarachant/pyRiemann
mne: https://github.com/mne-tools/mne-python

The SPD-Net part is originally from https://github.com/YirongMao/SPDNet

For subject-adaptive analysis, run `SPDNet_Federated_Transfer_Learning.py `

For subject-specific analysis, run `SPDNet_Local_Learning.py`

---

## Federated Transfer Learning Framework for Biomedical Applications

We developed a federated transfer learning framework for various biomedical applications.
The FTL framework adopts several federated learning frameworks such as FATE and pyTorch. 
It supports biomedical machine learning tasks with different types of data.
We believe the FTL framework provides an easy-to-learn tool for researchers to study machine learning tasks of 
different biomedical data with privacy protection and good performance. The FTL framework will be open-sourced soon. 

![Federated Transfer Learning Framework architecture](https://github.com/DashanGao/Federated-Transfer-Leraning-for-EEG/blob/master/imgs/ftl_framework.png)

---

For detailed information about our work, please refer to our paper published in EMBC 2020: 

[Federated Transfer Learning for EEG Signal Classification](https://arxiv.org/abs/2004.12321)

If this project helps you in your research, please cite our work in your paper.

```
@article{ju2020federated,
  title={Federated Transfer Learning for EEG Signal Classification},
  author={Ju, Ce and Gao, Dashan and Mane, Ravikiran and Tan, Ben and Liu, Yang and Guan, Cuntai},
  journal={IEEE Engineering in Medicine and Biology Society (EMBC)},
  year={2020}
}
```

---

This work is conducted by Hong Kong University of Science and Technology (HKUST), Southern University of Science and Technology (SUSTech), WeBank Co. Ltd. and Nanyang Technology University (NTU).

![Institution Logo](https://github.com/DashanGao/Federated-Transfer-Leraning-for-EEG/blob/master/imgs/institution_logo.png)

The authors are

![Authors](https://github.com/DashanGao/Federated-Transfer-Leraning-for-EEG/blob/master/imgs/authors_embc2020.png)


