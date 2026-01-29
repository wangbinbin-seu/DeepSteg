# DeepSteg: Deep learning based networks for image steganalysis and steganography
This is the open project for Deep learning based networks for image steganalysis and steganography, including our FaceAny, our DeepAny, and other reproduced literature on steganalysis and steganography.
## FaceAny: Fusion-aware Invertible Hiding Network for Multiple Face Anonymization [Paper Link]

### Overview
<img src="https://github.com/wangbinbin-seu/DeepSteg/blob/main/figures/faceany-architecture.png" width="600"/>
<img src="https://github.com/wangbinbin-seu/DeepSteg/blob/main/figures/faceany-result.jpg" width="600"/>


## DeepAny: Surface-guide Reversible Anonymization Network for Realistic Full-Body Privacy Preserving [Paper Link]
### Overview
<img src="https://github.com/wangbinbin-seu/DeepSteg/blob/main/figures/deepany-architecture.png" width="600"/>
<img src="https://github.com/wangbinbin-seu/DeepSteg/blob/main/figures/deepany-result.jpg" width="600"/>


#### The reproduced literature on steganalysis is listed as follows:
- XuNet (SPL2016): [**Structural Design of Convolutional Neural Networks for Steganalysis.**](https://ieeexplore.ieee.org/abstract/document/7444146) 
- YeNet (TIFS2017): [**Deep Learning Hierarchical Representations for Image Steganalysis.**](https://ieeexplore.ieee.org/abstract/document/7937836)
- StegNet (IH&MMSec2017): [**Fast and Effective Global Covariance Pooling Network for Image Steganalysis.**](https://dl.acm.org/doi/abs/10.1145/3335203.3335739)
- SRNet (TIFS2019): [**Deep Residual Network for Steganalysis of Digital Images.**](https://ieeexplore.ieee.org/abstract/document/8470101)
- ZhuNet (TIFS2020): [**Depth-Wise Separable Convolutions and Multi-Level Pooling for an Efficient Spatial CNN-Based Steganalysis.**](https://ieeexplore.ieee.org/abstract/document/8809687)
- SiaStegNet (TIFS2021): [**A Siamese CNN for Image Steganalysis.**](https://ieeexplore.ieee.org/document/9153041)
- More...

#### The reproduced papers on steganography is listed as follows:
- balujanet (TPAMI2020): [**Hiding images within images.**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8654686) 
- hidden (ECCV2018): [**HiDDeN: Hiding Data With Deep Networks.**](https://openaccess.thecvf.com/content_ECCV_2018/html/Jiren_Zhu_HiDDeN_Hiding_Data_ECCV_2018_paper.html)
- wengnet (ICMR2019): [**High-Capacity Convolutional Video Steganography with Temporal Residual Modeling.**](https://dl.acm.org/doi/abs/10.1145/3323873.3325011)
- hinet (ICCV2021): [**HiNet: Deep Image Hiding by Invertible Network.**](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html)
- More...

## Dependencies and Installation
- Python 3.8.13, PyTorch = 1.11.0
- Run the following commands in your terminal:

  `conda env create -f env.yaml`  

  `conda activate deepsteg`

## Updates
- ✅ 2025-01-14: reproduce the codes, models and results of DeepSteg.
- ✅ 2025-01-15: Release the first version of the project.
- ✅ 2025-01-29: Release the codes and models of FaceAny.
- **(To do)** More detail will be added ... 

## Overview
<img src="xxx.png" width="600"/>

**Benchmark results on BossBase**
| Model | Params(M) | Multi-Adds(G) | SRNet | ZhuNet | XuNet | YeNet | StegNet | SiaStegNet | 
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| balujanet |   |   |   |   |   |   |   |   |
| hidden |   |   |   |   |   |   |   |   |
| wengnet |   |   |   |   |   |   |   |   |
| hinet |   |   |   |   |   |   |   |   |

## Data Preparation
- The models are trained on the [DIV2K](https://opendatalab.com/DIV2K) training dataset, and the mini-batch size is set to 8, with half of the images randomly selected as the cover images and the remaining images as the secret images. 
- The trained models are tested on three test sets, including the DIV2K test dataset, 1000 images randomly selected from the ImageNet test dataset
- Here we provide [test sets](https://drive.google.com/file/d/1NYVWZXe0AjxdI5vuI2gF6_2hwoS1c4y7/view?usp=sharing).

- For train or test on the dataset,  e.g.  DIV2K, change the code in `config.py`:

    `line17:  data_dir = '' `
  
    `data_name_train = 'div2k'`
  
    `data_name_test = 'div2k'`
  
    `line30:  suffix = 'png' `

- Structure of the dataset directory:

<center>
  <img src=https://github.com/albblgb/pusnet/blob/main/utils/dataset_folder_structure.png width=36% />
</center>

## Get Started
#### Training for steganalysis
1. Change the code in `config.py`

    `line4: mode = 'train'`
   
    `line17: train_data_dir = ''`
   
    `line18: val_data_dir = ''`

    `line20: stego_img_height = `
   
    `line21: stego_img_channel = `

3. Run `python *net.py`. For example, `python srnet.py`

#### Training for steganography
1. Change the code in `config.py`

    `line4:  mode = 'train' ` 

2. Run `python *net.py`, for example, `python wengnet.py`


#### Testing for steganalysis
1. Change the code in `config.py`

    `line4: mode = 'test' `

    `line19: test_data_dir = ''`
  
    `line36-41: pre_trained_*net_path = ''`

3. Run `python *net.py`

- The trained steganalysis networks will be saved in 'checkpoint/'
- The results and running logs will be saved in 'results/'

#### Testing for steganography
1. Change the code in `config.py`

    `line4:  mode = 'test' `
  
    `line36-41:  test_*net_path = '' `

2. Run `python *net.py`

- Here we provide [trained models](https://drive.google.com/drive/folders/1lM9ED7uzWYeznXSWKg4mgf7Xc7wjjm8Q?usp=sharing).
- The processed images, such as stego image and recovered secret image, will be saved at 'results/images'
- The training or testing log will be saved at 'results/*.log'
 
## Others
- We don't adopt the default settings from the literature. Instead, all stegeanalysis networks are optimized using Adam solver with a weight decay of 1e-5 and an initial learning rate of 2e-4.

## Results
The inference results on benchmark datasets are available at
[Google Drive](https://drive.google.com/drive/foldersXXXX) or [Baidu Netdisk](https://pan.baidu.com/XXXX) (access code: XXXX).


## Acknowledgement
- We would like to express our gratitude to the authors of the relevant papers. If you find this work useful, please cite the original paper source.
- We would like to express our gratitude to the listed open-source projects.
  - [Deep-Steganalysis](https://github.com/albblgb/Deep-Steganalysis.git)
  - [Hiding-images-within-images](https://github.com/albblgb/Hiding-images-within-images.git)
## Contact
- If you have any question, please email yew.wang.os@hotmail.com
