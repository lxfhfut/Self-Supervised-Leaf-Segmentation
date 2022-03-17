# Self-supervised-leaf-segmentation
## Introduction
This repository contains the code for *Self-supervised Leaf Segmentation Under Complex Lighting Conditions*. Intended for growth monitoring in a real-world cannabis greenhouse, this project leverages self-supervised learning for effective and generalizable leaf segmentation in images taken under artificial grow lights, without resorting to any annotated data.
## Datasets
We use two datasets in our experiments: Our Cannabis dataset and the CVPPP leaf segmentation challenge (LSC) dataset. Our Cannabis dataset contains 120 images captured under three different lighting conditions: "Natural", "Yellow", and "Purple". To simulate the "Yellow" and "Purple" lighting conditions for the CVPPP dataset, we generate the "Yellow" and "Purple" versions of each image by manipulating the hue value of each pixel. The original CVPPP LSC dataset is refered to as "Natural". Sample images are shown below:

<figure>
  <img src="images/2021_07_03_01_rgb.png" height="200" width="200"/> 
<figcaption align = "center"><b>Cannabis "Natural"</b></figcaption>
  <img src="images/2021_07_09_09_rgb.png" height="200" width="200"/>
  <figcaption align = "center"><b>Cannabis "Natural"</b></figcaption>
</figure>
<img src="images/2021_07_11_07_rgb.png" height="250" width="250"/> 

## Demos
### Cannabis leaves
<img src="images/2021_06_30_04_result.gif" width="400"/> <img src="images/2021_07_04_02_result.gif" width="400"/> 
### Small-sized leaves in the CVPPP LSC dataset
<img src="images/plant042_result.gif" width="400"/> <img src="images/plant0868_result.gif" width="400"/>
### Medium-sized leaves in the CVPPP LSC dataset
<img src="images/plant0906_result.gif" width="400"/> <img src="images/plant0946_result.gif" width="400"/>
### Large-sized leaves in the CVPPP LSC dataset
<img src="images/plant030_result.gif" width="400"/> <img src="images/plant158_result.gif" width="400"/>
