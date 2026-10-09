## DLRL: Dual-Level Reliability Learning for Multimodal Remote Sensing Clustering

Python implementation for our paper "DLRL: Dual-Level Reliability Learning for Multimodal Remote Sensing Clustering".

## Introduction

A Python implementation of the multimodal remote sensing clustering framework presented in:

   <b><i> 'DLRL: Dual-Level Reliability Learning for Multimodal Remote Sensing Clustering.' </i></b>

DLRL learns sample-specific modality weights from cross-modal context to suppress unreliable features before shared Transformer encoding. An exponential moving average teacher estimates inter-sample affinity from two augmented views, and their geometric agreement provides soft multi-positive supervision for contrastive learning. The framework combines reliability-aware fusion and contrastive learning with prototype-based clustering and EMA anchoring.

## Project Structure

The project is organized as follows:

```text
DLRL/
├── modules/
├── Toolbox/
├── utils/
├── config.yaml
├── config1.yaml
├── config2.yaml
├── requirements.txt
└── train.py
```
## Requirements

* torch
* torchvision
* numpy
* scipy
* scikit-learn
* PyYAML
* spectral
* einops
* munkres
* scikit-image
* matplotlib

## More

For more related researches, please visit my homepage: https://dumingjing.github.io/. For data and discussion, please message Mingjing Du (杜明晶@江苏师范大学): dumj@jsnu.edu.cn.