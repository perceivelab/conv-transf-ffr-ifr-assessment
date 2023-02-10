<div align="center">

# A Hybrid Convolutional-Transformer Model for FFR and iFR Assessment from Angiography Data
Raffaele Mineo, Federica Proietto Salanitri, Ovidio De Filippo, Michele Millesimo, Gaetano Maria De Ferrari, Marco Aldinucci, Daniela
Giordano, Fabrizio D’Ascenzo, Simone Palazzo, Concetto Spampinato

<!---[![Paper](http://img.shields.io/badge/paper-arxiv.2206.10048-B31B1B.svg)]()-->

</div>

# Overview
Official PyTorch implementation of paper: <b>"A Hybrid Convolutional-Transformer Model for FFR and iFR Assessment from Angiography Data"</b>

# Abstract
The assessment of coronary artery stenosis from X-ray angiography imaging is a challenging task, due to the variegate appearance of stenoses, the presence of overlapping and occluding vessels and the relative small size of the affected image region. Current approaches in the literature tackle the problem through a combination of slice-oriented 2D models, multi-view inputs and/or key frame information. In this paper, we propose a deep learning classification model for the assessment of angiography-based non-invasive fractional flow-reserve (FFR) and instantaneous wave-free ratio (iFR) of intermediate coronary stenosis. Our approach predicts whether a coronary artery stenosis is hemodynamically significant or not, and provides direct FFR and iFR estimates, by taking advantage of 3D CNN inductive bias, to quickly learn local spatio-temporal features, and attention-based transformer layers, to capture long-range relations within the feature volume. The proposed method achieves state-of-the-art performance (on a variety of metrics) on a dataset of 778 exams from 389 patients. Moreover, unlike existing methods, our approach employs only a single angiography view and does not require knowledge of the key frame; supervision at training time is provided by a combination of a classification loss (based on a threshold of the FFR/iFR values) and a regression loss for direct estimation.
Finally, the analysis of model interpretability and calibration shows that, in spite of the complexity of angiographic imaging data, our method is able to robustly identify the location of the stenosis, as well as to correlate prediction uncertainty to the provided output scores.

# Method
 <p align = "center"><img src="img/net.png" width="600" style = "text-align:center"/></p>
 
 # How to run
 The code expects ...:
 ```python

```
 
 ## Pre-requisites:
- NVIDIA GPU (Tested on Nvidia Tesla T4 GPUs )
- [Requirements](env.yaml)
- PyTorch >= 
- MONAI >= 

## Train Example
```bash
python 

```

## Notes
The pre-trained backbone weights can be downloaded from these links:
- [I3D](https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt)
- [S3D](https://drive.google.com/uc?export=download&id=1HJVDBOQpnTMDVUM3SsXLy0HUkf_wryGO)
- [SwinTransformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth)
