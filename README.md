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
 <p align = "center"><img src="img/Net.png" width="900" style = "text-align:center"/></p>
Architecture of the proposed approach for stenosis significance assessment. Input angiography videos are first processed by a pre-trained 3D convolutional model for local feature extraction. Then, self-attention layers based on transformers are employed to capture intra-relations in space and time, and the resulting intermediate representation is fed to three output branches, providing predictions as either a significance class or a regression estimate of FFR/iFR values.

# Interpretability of results
Interpretability maps, computed by M3D-cam, of our method (last row) and other state-of-the-art approaches. The red bounding box highlights the location of the major stenosis as identified by cardiologists. 

<p align = "center"><img src="img/interpretability.png" width="600" style = "text-align:center"/></p>

# How to run
The code expects a JSON file in the format support by MONAI, passed via the `--split_path` argument, with the following structure:
```
{
 "num_fold": <N>, 
 "fold0": { "train": [ {"image": <path>,
                        "label": <class>,
                        "FFR": <value>,
                        "iFR": <value>},
                       ...
                       {"image": <path>,
                        "label": <class>,
                        "FFR": <value>,
                        "iFR": <value>}
                      ],
            "val": [ {"image": <path>,
                        "label": <class>,
                        "FFR": <value>,
                        "iFR": <value>},
                      ...],
            "test": [{"image": <path>,
                        "label": <class>,
                        "FFR": <value>,
                        "iFR": <value>},
                      ...]}, 
 "fold1": { "train": [...],
            "val": [...],
            "test": [...]},
 ...
}
```
`<N>`, `<path>`, `<class>` and `<value>` fields should be filled as appropriate.
Each path should point to a `.npy`file containing a 3D (2D+T) tensor, representing an angiography video.

## Pre-requisites:
- NVIDIA GPU (Tested on Nvidia Tesla T4 GPUs )
- [Requirements](requirements.txt)

## Train Example
 To start training, simply run (using default arguments):
 
 ```python train.py --root_dir='<dataset_path>' --split_path='<split_json_path>'```
 
To start distributed training, use:

```
python -m torch.distributed.launch --nproc_per_node=<N_GPUS> --use_env train.py --root_dir='<dataset_path>' --split_path='<split_json_path>'
```

## Test Example
To start evaluation, simply run (using default arguments):

```python test.py --logdir='<log_path>' --start_tornado_server=1 --enable_explainability=1```

Log directories are automatically created upon training inside a `logs` directory.

To start distributed testing, use:

```
python -m torch.distributed.launch --nproc_per_node=<N_GPUS> --use_env test.py --logdir='<log_path>' --start_tornado_server=1 --enable_explainability=1
```

<!--- ## Notes --->
<!--# Acknowledgements
This code is taken from https://github.com/IngRaffaeleMineo/3D-BCPTcode and modified to our purposes.-->
