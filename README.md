# MedSAM3: Delving into Segment Anything with Medical Concepts

[//]: # (<div align="center">)

[//]: # ()
[//]: # (**Anglin Liu**<sup>1,*</sup>, **Rundong Xue**<sup>2,*</sup>, **Xu R. Cao**<sup>3,†</sup>, **Yifan Shen**<sup>3</sup>, **Yi Lu**<sup>1</sup>, **Xiang Li**<sup>3</sup>, **Qianqian Chen**<sup>4</sup>, **Jintai Chen**<sup>1,5,†</sup>)

[//]: # ()
[//]: # (<sup>1</sup> The Hong Kong University of Science and Technology &#40;Guangzhou&#41;  )

[//]: # (<sup>2</sup> Xi’an Jiaotong University  )

[//]: # (<sup>3</sup> University of Illinois Urbana-Champaign  )

[//]: # (<sup>4</sup> Southeast University  )

[//]: # (<sup>5</sup> The Hong Kong University of Science and Technology  )

[//]: # ()
[//]: # (<small><sup>*</sup> Equal Contribution &nbsp;&nbsp; <sup>†</sup> Corresponding Author</small>)

[//]: # ()
[//]: # ([![arXiv]&#40;https://img.shields.io/badge/arXiv-2511.19046-b31b1b.svg?logo=arxiv&#41;]&#40;https://arxiv.org/abs/2511.19046&#41;)

[//]: # (&nbsp;)

[//]: # ([![Hugging Face]&#40;https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-ffd21e&#41;]&#40;https://huggingface.co/lal-Joey/MedSAM3_v1&#41;)

[//]: # ()
[//]: # (</div>)

[//]: # (**We will continuously update the documentation and examples to optimize this repository.**)

---

## 📖 Introduction

**Med-SAMUS** is a pure text-guided (concept-guided) medical image segmentation model. Unlike traditional models that rely on bounding boxes or points, MedSAM3 leverages specific medical concepts to segment targets across a wide range of modalities.

### 🌟 Key Features & Dataset Statistics

(1)[Dataset B:](https://helward.mmu.ac.uk/STAFF/M.Yap/dataset.php) M. H. Yap et al., Breast ultrasound region of interest detection and lesion localisation, Artif. Intell. Med., vol. 107, no. August 2019, p. 101880, 2020.  

(2)[STU:](https://github.com/xbhlk/STU-Hospital.git) Z. Zhuang, N. Li, A. N. Joseph Raj, V. G. V Mahesh, and S. Qiu, “An RDAU-NET model for lesion segmentation in breast ultrasound images,” PLoS One, vol. 14, no. 8, p. e0221535, 2019.


## 📦 Model & Weights

We adopted a parameter-efficient fine-tuning strategy based on **SAM3** using **LoRA (Low-Rank Adaptation)**.

We are releasing our first version (**v1**) of the LoRA weights.

| Model Version  | Base Model | Method | Link |
|:---------------| :--- | :--- | :--- |
| **MedSAM3-v1** | SAM3 | LoRA Fine-tuning | [**Download LoRA Weights**](https://huggingface.co/lal-Joey/MedSAM3_v1) |
| **Med-SAMUS**  | SAM3 | LoRA Fine-tuning | [**Download LoRA Weights**](https://huggingface.co/lal-Joey/MedSAM3_v1) |

## 🔗 References

This project is built upon the following excellent open-source projects. Please refer to them for the base environment setup. If you encounter code-related issues, please also refer to the specific instructions and documentation provided by these works:

* **SAM3:** [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)
* **SAM3_LoRA:** [https://github.com/Sompote/SAM3_LoRA](https://github.com/Sompote/SAM3_LoRA)

## 🚀 Inference

Follow these steps to run inference on your medical images.

### 1. Setup
```python
# Clone repository
git clone https://github.com/Joey-S-Liu/MedSAM3.git
cd MedSAM3

# Install dependencies
pip install -e .

# Login to Hugging Face
hf auth login
# Paste your token when prompted
```

### 2. Inference Code
```python
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --prompt "skin lesion" \
  --threshold 0.5 \
  --nms-iou 0.5 \
  --output skin_lesion.png
```

### 3. Training Code
```python
python3 train.py --config configs/full_lora_config.yaml
```

## 📧 Contact

If you have any questions regarding this project, please feel free to contact the corresponding authors:


## 🖊️ Citation

