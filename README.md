# Self-supervised-leaf-segmentation
## Introduction
This repository contains the code for [*Self-supervised Leaf Segmentation Under Complex Lighting Conditions*](https://www.sciencedirect.com/science/article/abs/pii/S0031320322005015). Intended for growth monitoring in a real-world smart greenhouse environment, this project leverages self-supervised learning for effective and generalizable leaf segmentation in images taken under artificial grow lights, without resorting to any annotated data. If you find this work useful, please cite our paper:
```
@article{lin2022self,
  title={Self-Supervised Leaf Segmentation under Complex Lighting Conditions},
  author={Lin, Xufeng and Li, Chang-Tsun and Adams, Scott and Kouzani, Abbas and others},
  journal={Pattern Recognition},
  pages={109021},
  year={2022},
  publisher={Elsevier}
}
```
## Instructions
The code was only tested on Ubuntu 20.04 with an NVIDIA GeForce RTX 2080 Ti. To get started, make sure the dependencies are installed via Anaconda:
```
# create and activate environment
conda env create -f ssls.yml
conda activate ssls
```
## Datasets
We use two datasets in our experiments: **Our Cannabis dataset** and the **CVPPP leaf segmentation challenge (LSC) dataset**. 
These two datasets and the pretained color correction models can be downloaded <a href="https://drive.google.com/drive/folders/1tmaRUmdnDhyvnznOWD_S1sYkxb-g02MT?usp=sharing" target="_blank">here</a>. Put the downloaded 'pretrained' folder in the root directory of the source code (i.e., at the same level as folders 'imgs', 'exmaples' and 'output').

Our Cannabis dataset contains 120 images captured under three different lighting conditions: "Natural", "Yellow", and "Purple", with 40 images obtained in each lighting condition. 

Cannabis "Natural"             | Cannabis "Yellow"         | Cannabis "Purple"
:-------------------------:|:-------------------------:|:-------------------------:
<img src="examples/2021_07_03_01_rgb.png" height="200" width="200"/>  |  <img src="examples/2021_07_09_09_rgb.png" height="200" width="200"/>  |. <img src="examples/2021_07_11_07_rgb.png" height="200" width="200"/> 

To simulate the "Yellow" and "Purple" lighting conditions for the CVPPP dataset, we generate the "Yellow" and "Purple" versions of each image by manipulating the hue value of each pixel. The original CVPPP LSC dataset is refered to as "Natural".

CVPPP LSC "Natural"             | CVPPP LSC "Yellow"         | CVPPP LSC "Purple"
:-------------------------:|:-------------------------:|:-------------------------:
<img src="examples/plant066_rgb.png" height="200" width="200"/>  |  <img src="examples/plant025_rgb.png" height="200" width="200"/>  |. <img src="examples/plant035_rgb.png" height="200" width="200"/> 

## Demos
### Cannabis leaves
<img src="examples/2021_06_30_04_result.gif" width="400"/> <img src="examples/2021_07_04_02_result.gif" width="400"/> 
### Small-sized leaves in the CVPPP LSC dataset
<img src="examples/plant042_result.gif" width="400"/> <img src="examples/plant0868_result.gif" width="400"/>
### Medium-sized leaves in the CVPPP LSC dataset
<img src="examples/plant0906_result.gif" width="400"/> <img src="examples/plant0946_result.gif" width="400"/>
### Large-sized leaves in the CVPPP LSC dataset
<img src="examples/plant030_result.gif" width="400"/> <img src="examples/plant158_result.gif" width="400"/>
