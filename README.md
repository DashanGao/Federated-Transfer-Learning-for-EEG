# Federated Transfer Learning For EEG Signal Classification

Authors: Ce Ju, Dashan Gao, Ravikiran Mane, Ben Tan, Yang Liu and Cuntai Guan

Published in: 42nd Annual International Conferences of the IEEE Engineering in Medicine and Biology Society in 
conjunction with the 43rd Annual Conference of the Canadian Medical and Biological Engineering Society (EMBS)
July 20-24, 2020 via the EMBS Virtual Academy

---

## Introduction

<!--- ![Federated Learning](https://github.com/DashanGao/Federated-Transfer-Leraning-for-EEG/blob/master/imgs/federated_learning.png =250*250)![Federated Learning EEG](https://github.com/DashanGao/Federated-Transfer-Leraning-for-EEG/blob/master/imgs/federated_learning_eeg.png =250*250) --->

The impact of deep learning (DL) methodologies within the sphere of Brain-Computer Interfaces (BCI) for the categorization of electroencephalographic (EEG) transcriptions has been stymied by the dearth of expansive datasets. Constraints linked to privacy concerns surrounding EEG signals impede the creation of large EEG-BCI datasets through the amalgamation of various smaller datasets for the shared training of the machine learning model. Consequently, this paper presents a novel privacy-preserving DL framework named federated transfer learning (FTL) for EEG categorization, which is predicated on the federated learning structure. Utilizing the single-trial covariance matrix, this proposed framework extracts common discriminative information from multi-subject EEG data via domain adaptation techniques. The effectiveness of the proposed framework is assessed against the PhysioNet dataset for a two-class motor imagery classification. All while circumventing direct data sharing, our FTL method attains a 2% improvement in classification accuracy in a subject-adaptive analysis. Moreover, when multi-subject data is not available, our framework delivers a 6% increase in accuracy in comparison to other leading-edge DL frameworks.

---

## Network Architecture


Our proposed architecture comprises the following four layers: the manifold reduction layer, the common embedded space, the tangent projection layer, and the federated layer. The function of each layer is detailed below:

1. **Manifold Reduction Layer**: Spatial covariance matrices are consistently presumed to inhabit high-dimensional Symmetric Positive Definite (SPD) manifolds. This layer acts as a linear map from the high-dimensional SPD manifold to the low-dimensional counterpart, with undefined weights reserved for learning.

2. **Common Embedded Space**: The common space is the low-dimensional SPD manifold, whose elements are diminished from each high-dimensional SPD manifold. It is specifically designed for the transfer learning setting.

3. **Tangent Projection Layer**: The function of this layer is to project the matrices on SPD manifolds to its tangent space, which is a local linear approximation of the curved space.

4. **Federated Layer**: Deep neural networks are implemented in this layer. For the transfer learning setting, the parameters of neural networks are updated by federated aggregation.


---

### How To Run The Code

We extend our gratitude to the open-source community, which facilitates the wider dissemination of the work of other researchers as well as our own. The coding style in this repo is relatively rough. We welcome anyone to refactor it to make it more effective. The codebase for our models builds heavily on the following repositories:
[<img src="https://img.shields.io/badge/GitHub-pyRiemann-b31b1b"></img>](https://github.com/pyRiemann/pyRiemann) 
[<img src="https://img.shields.io/badge/GitHub-SPDNet(Z.W.Huang)-b31b1b"></img>](https://github.com/zhiwu-huang/SPDNet)
[<img src="https://img.shields.io/badge/GitHub-mne-b31b1b"></img>](https://github.com/mne-tools/mne-python)

### Data pre-processing

Please put your training data and labels into a directory "raw_data/" in this project.
The package `mne` is adopted for EEG data pro-processing. To generate the required data as SPDNet input, please refer to the following example code: 

```python        
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci

# Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 1., 2.
event_id = dict(hands=2, feet=3)
subject = 7
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_files = [
    read_raw_edf(f, preload=True) for f in eegbci.load_data(subject, runs)
]
raw = concatenate_raws(raw_files)

picks = pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
# subsample elecs
picks = picks[::2]

# Apply band-pass filter
raw.filter(7., 35., method='iir', picks=picks)

events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
    verbose=False)
labels = epochs.events[:, -1] - 2

# cross validation
cv = KFold(n_splits=10, random_state=42)
# get epochs
epochs_data_train = 1e6 * epochs.get_data()

# compute covariance matrices
cov_data_train = Covariances().transform(epochs_data_train)
```

### Model training

For subject-adaptive analysis, run `SPDNet_Federated_Transfer_Learning.py `

For subject-specific analysis, run `SPDNet_Local_Learning.py`

---

## Federated Transfer Learning Framework for Biomedical Applications

We have engineered a Federated Transfer Learning (FTL) framework designed for a range of biomedical applications. The FTL framework incorporates various federated learning architectures including FATE and PyTorch, and it caters to biomedical machine learning tasks involving diverse types of data. We posit that the FTL framework offers a user-friendly tool for researchers to explore machine learning tasks pertaining to various biomedical data types, while ensuring privacy protection and superior performance. We intend to open-source the FTL framework in the near future.

![Federated Transfer Learning Framework architecture](https://github.com/DashanGao/Federated-Transfer-Leraning-for-EEG/blob/master/imgs/ftl_framework.png)

---

## Contributions

We implement a federated learning framework to construct BCI models from multiple subjects with heterogeneous distributions. Key advantages of our Federated Transfer Learning (FTL) approach include:

1. **Privacy**: FTL preserves user privacy by retaining the EEG data of each subject on-device, preventing any potential data leaks.
2. **Spatial Covariance Matrix Utilization**: By using the spatial covariance matrix as an input, FTL surpasses other state-of-the-art deep learning methods in EEG-MI tasks, resulting in a 6% increase in accuracy.
3. **Transfer Learning**: FTL leverages transfer learning to attain superior classification accuracy, even for subjects whose EEG signals might be considered 'bad' or challenging to interpret.

The major contribution of this study lies in reporting promising trial results within specific scenarios. However, due to the relatively small number of trials per subject in the test dataset, the classification results exhibit considerable randomness. Under varying datasets, tasks, and division methods, our approach will yield differing performances. Hence, we suggest users to fine-tune parameters and network structures according to their own contexts in order to attain optimal results.

---


## Cite Us

For an in-depth insight into our work, we kindly direct you to our paper presented at the 42nd Annual International Conferences of the IEEE Engineering in Medicine and Biology Society, held in conjunction with the 43rd Annual Conference of the Canadian Medical and Biological Engineering Society (EMBS), which took place between July 20-24, 2020, through the EMBS Virtual Academy:

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

## Follow-up Works

The fundamental neural network structure currently utilized for transfer learning is a second-order neural network structure. In our follow-up work, we further developed this network structure, proposing several geometric BCI classifiers. If you are interested in our follow-up work, please proceed to the following URL:https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet These geometric BCI classifiers, inspired by differential geometry, have achieved state-of-the-art results in subject-specific scenarios across multiple motor imagery datasets.

---

## Authors

This research was undertaken by a collaborative team from the Hong Kong University of Science and Technology, the Southern University of Science and Technology, WeBank Co. Ltd., and Nanyang Technological University.

![Institution Logo](https://github.com/DashanGao/Federated-Transfer-Leraning-for-EEG/blob/master/imgs/institution_logo.png)

The authors are

![Authors](https://github.com/DashanGao/Federated-Transfer-Leraning-for-EEG/blob/master/imgs/authors_embc.png)


