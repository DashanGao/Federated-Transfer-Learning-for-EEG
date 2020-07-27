# Federated Transfer Learning For EEG Signal Classification
Authors: Ce Ju, Dashan Gao, Ravikiran Mane, Ben Tan, Yang Liu and Cuntai Guan

---

## Introduction

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



Please install pyRiemann and mne

pyRiemann: https://github.com/alexandrebarachant/pyRiemann
mne: https://github.com/mne-tools/mne-python

The SPD-Net part is original from https://github.com/YirongMao/SPDNet

For subject-adaptive analysis, run FTL_Transfer.py 
For subject-specific analysis, run FTL_NonTransfer.py

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