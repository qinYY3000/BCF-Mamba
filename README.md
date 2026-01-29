# BC-Mamba: Boundary-Aware Contextual CNNs-Mamba for Accurate Ultrasound Image Segmentation 🚀

Official repository for: *[BCF-Mamba: Boundary-Aware Contextual CNN–Mamba Fusion for Ultrasound Image Segmentation](https://github.com/qinYY3000/BCF-Mamba)*


The source code will be released to the public in the near future.

If you have any questions, please contact: chen_wenqin2002@163.com

### Ultrasound dataset:
(1)[BUSI:](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) W. Al-Dhabyani., Dataset of breast ultrasound images, Data Br. 28 (2020) 104863.  

(2)[Dataset B:](https://helward.mmu.ac.uk/STAFF/M.Yap/dataset.php) M. H. Yap et al., Breast ultrasound region of interest detection and lesion localisation, Artif. Intell. Med., vol. 107, no. August 2019, p. 101880, 2020.  

(3)[STU:](https://github.com/xbhlk/STU-Hospital.git) Z. Zhuang, N. Li, A. N. Joseph Raj, V. G. V Mahesh, and S. Qiu, “An RDAU-NET model for lesion segmentation in breast ultrasound images,” PLoS One, vol. 14, no. 8, p. e0221535, 2019.

(4)[DDTI:](https://drive.google.com/file/d/1wwlsEhwfSyvQsJBRjeDLhUjqZh8eaH2R/view) L. Pedraza, C. Vargas, F. Narvaez, O. Duran, E. Munoz, and E. Romero,  “An open access thyroid ultrasound image database,” in 10th International symposium on medical information processing and analysis, vol. 9287, pp. 188–193, SPIE, 2015.

(5)[SZU-BCH-TUS983:](https://github.com/qinYY3000/BCF-Mamba)

## Main Results

- BUSI
- Dataset B
- STU
- DDTI
- SZU-BCH-TUS983

## Installation

**Step-1:** Create a new conda environment & install requirements

```shell
conda create -n bcmamba python=3.10
conda activate bcmamba

cd selective_scan
pip install .

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0   (cuda 11.8)
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.2.4
pip install torchinfo timm numba
```

**ImageNet pretrained model:** 

We use the ImageNet pretrained VMamba-Tiny model from [VMamba](https://github.com/MzeroMiko/VMamba). You need to download the model checkpoint and put it into `pretrained_ckpt/vmamba_tiny_e292.pth`


We use the ImageNet pretrained ConvNeXt-T model from [ConvNeXt](https://github.com/facebookresearch/ConvNeXt). You need to download the model checkpoint and put it into `pretrained_ckpt/convnext_tiny_1k_224_ema.pth`

```
wget https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmtiny_dp01_ckpt_epoch_292.pth
mv vssmtiny_dp01_ckpt_epoch_292.pth data/pretrained/vmamba/vmamba_tiny_e292.pth
```


## Acknowledgements

We thank the authors of [Mamba](https://github.com/state-spaces/mamba), [VMamba](https://github.com/MzeroMiko/VMamba), [Swin-UMamba](https://github.com/JiarunLiu/Swin-UMamba) and [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) for making their valuable code & data publicly available.


## Citation

```

```



