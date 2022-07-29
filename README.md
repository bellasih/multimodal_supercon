# Multimodal SuperCon: Classifier for Drivers of Deforestation in Indonesia

This repository contains code implementations of contrastive learning architecture, called Multimodal SuperCon, for classifying drivers of deforestation in Indonesia using satellite images obtained from Landsat 8. Multimodal SuperCon is an architecture which combines contrastive learning and multimodal fusion to handle the available deforestation dataset.

This project is using several papers as the main references:
1. [ForestNet](https://arxiv.org/abs/2011.05479)
2. [Rotation Equivariant Deforestation Segmentation and Driver Classification](https://arxiv.org/abs/2110.13097)

## Architecture
This project implements two-stage learning, representation and classification stage for training the models. Training process takes 2 step:
1. Representation Stage using Supervised Contrastive Learning.
2. Classification Stage using Supervised Learning with Multimodal Fusion.

## How to Use
### Requirements:
Main libraries and dependencies:
1. `PyTorch`
1. `Shapely`: python package for set-theoretic analysis and manipulation of planar features, beneficial for spatial data analysis
1. `Albumentation`: python package for image augmentations
### Run The Program
1. To run the program, simply by re-running the available notebook: `Training - Effnet + Resnet.ipynb` and `Training - UNet.ipynb`
1. If you want to add any available auxiliaries/predictors from ForestNet dataset, you can modify the backbone model where the code implementation can be found under `model` folder (will update the other examples, especially with four auxiliaries/predictors, soon)

## How to Cite
### Bibtex

```
will update the bibtex soon
```