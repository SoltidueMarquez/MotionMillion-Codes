# Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data

[Ke Fan](https://vankouf.github.io/)<sup>1</sup>, [Shunlin Lu](https://shunlinlu.github.io/)<sup>2</sup>, [Minyue Dai](https://jixiii.github.io/)<sup>3</sup>, [Runyi Yu](https://ingrid789.github.io/IngridYu/)<sup>4</sup>, [Lixing Xiao](https://li-xingxiao.github.io/homepage/)<sup>5</sup>, [Zhiyang Dou](https://frank-zy-dou.github.io/)<sup>6</sup>, [Junting Dong](https://jtdong.com/)<sup>7</sup>, [Lizhuang Ma](https://scholar.google.com/citations?user=yd58y_0AAAAJ&hl=zh-CN)<sup>1,8</sup>, [Jingbo Wang](https://wangjingbo1219.github.io/)<sup>7</sup>

Shanghai Jiao Tong University<sup>1</sup>, The Chinese University of Hong Kong, Shenzhen<sup>2</sup>, Fudan University<sup>3</sup>, Hong Kong University of Science and Technology<sup>4</sup>, Zhejiang University<sup>5</sup>, The University of Hong Kong<sup>6</sup>, Shanghai AI Laboratory<sup>7</sup>, East China Normal University<sup>8</sup>.


<p align="center">
  <a href=''>
  <img src='https://img.shields.io/badge/Arxiv-2404.19759-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href=''>
  <img src='https://img.shields.io/badge/Paper-PDF-purple?style=flat&logo=arXiv&logoColor=yellow'></a> 
  <a href=''>
  <img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
  <a href=''>
  <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'></a> 
  <a href=''>
  <img src='https://img.shields.io/badge/Bilibili-Video-4EABE6?style=flat&logo=Bilibili&logoColor=4EABE6'></a>
  <a href=''>
  <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
</p>


![](assets/teaser.jpg)

## 🤩 Abstract

> Generating diverse and natural human motion sequences based on textual descriptions constitutes a fundamental and challenging research area within the domains of computer vision, graphics, and robotics. Despite significant advancements in this field, current methodologies often face challenges regarding zero-shot generalization capabilities, largely attributable to the limited size of training datasets. Moreover, the lack of a comprehensive evaluation framework impedes the advancement of this task by failing to identify directions for improvement. In this work, we aim to push text-to-motion into a new era, that is, to achieve the generalization ability of zero-shot. To this end, firstly, we develop an efficient annotation pipeline and introduce MotionMillion—the largest human motion dataset to date, featuring over 2,000 hours and 2 million high-quality motion sequences. Additionally, we propose MotionMillion-Eval, the most comprehensive benchmark for evaluating zero-shot motion generation. Leveraging a scalable architecture, we scale our model to 7B parameters and validate its performance on MotionMillion-Eval. Our results demonstrate strong generalization to out-of-domain and complex compositional motions, marking a significant step toward zero-shot human motion generation.

<!-- ## 🤼‍♂ Arena -->

## 📢 News
- **[TODO]** Release Dataset.
- **[2025/07/03]** Train code, Inference code and Model checkpoints are released.
- **[2025/06/26]** MotionMillion is officially accepted by **ICCV 2025**.

## 👨‍🏫 Quick Start

This section provides a quick start guide to set up the environment and run the demo. The following steps will guide you through the installation of the required dependencies, downloading the pretrained models, and preparing the datasets. 

<details>
  <summary><b> 1. Conda environment </b></summary>

```
conda create python=3.8.11 --name motionmillion
conda activate motionmillion
```

Install the packages in `requirements.txt`.

```
pip install -r requirements.txt
```

We test our code on Python 3.8.11 and PyTorch 2.4.1.

</details>

<details>
  <summary><b> 2. Dependencies </b></summary>

<!-- <details> -->
  <summary><b>🥳  Run the following command to install git-lfs</b></summary>

```
conda install conda-forge::git-lfs
```

<!-- </details> -->

<!-- <details> -->
  <summary><b>🤖 Download SMPL+H and DMPL model</b></summary>

  1. Download [SMPL+H](https://mano.is.tue.mpg.de/download.php) (Extended SMPL+H model used in AMASS project)
  2. Download [DMPL](https://smpl.is.tue.mpg.de/download.php) (DMPLs compatible with SMPL)
  3. Place all models under `./body_models/`
<!-- </details> -->

<!-- <details> -->
<summary><b>👤 Download human model files</b></summary>

1. Download files from [Google Drive](https://drive.google.com/file/d/1y5jthVfCcMkT4cPNlyctH_AMDNz48e43/view?usp=sharing)
2. Place under `./body_models/`
<!-- </details> -->

<!-- <details> -->
<summary><b>⚙️ Run the script to download dependencies materials:</b></summary>

```
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators_on_motionmillion.sh
bash prepare/download_T5-XL.sh
```
<!-- </details> -->

</details>

<details>
  <summary><b> 3. Pretrained models </b></summary>

We provide our 3B and 7B models trained on train.txt and all.txt respectively. Our 7B-all achieves the best zero-shot performance. Run the script to download the pre-trained models:

```
bash prepare/download_pretrained_models.sh
```

</details>


<details>
  <summary><b> 4. Prepare the datasets </b></summary>
  Comming Soon!
  The dataset structure will be like:

```
dataset
├── MotionMillion
│   ├── motion_data
│   │   └── vector_272
│   │       ├── ...
│   │       └── ...
│   ├── texts
│   │   ├── ...
│   │   └── ...
│   │── mean_std
│   │    └── vector_272
│   │        ├── mean.npy
│   │        └── std.npy
│   │── split
│   │   └── version1
│   │       ├── t2m_60_300
│   │       │   ├── train.txt
│   │       │   ├── test.txt
│   │       │   ├── val.txt
│   │       │   └── all.txt
│   │       └── tokenizer_96
│   │       │   ├── train.txt
│   │       │   ├── test.txt
│   │       │   └── val.txt
├── ...

```
</details>


## 🎬 Inference

Please make sure that you have finished the preparations in Quick Start.

If you want to test the text-to-motion inference by yourself, please run the following commands:

```
bash scripts/inference/single_inference/test_t2m_3B.sh
bash scripts/inference/single_inference/test_t2m_7B.sh
```
please remind to replace the `${resume-pth}` and the `${resume-trans}` to the real path of your tokenizer and t2m model.

If you want to test our MotionMillion-Eval benchmark, please run the following commands:

```
bash scripts/inference/batch_inference/test_t2m_3B.sh
bash scripts/inference/batch_inference/test_t2m_7B.sh
```

The MotionMillion-Eval prompts are save in assets/infer_batch_prompt.


## 🚀 Train your own models

We provide the training guidance for motion reconstruction and text-to-motion tasks. The following steps will guide you through the training process.

<details>
  <summary><b> 2. Train Tokenizer </b></summary>

For multi-gpus: run the following command: (We train our tokenizer by 4gpus on 80G gpu.)

```
bash scripts/train/train_tokenizer.sh
```

For single: run the following command:

```
bash scripts/train/train_tokenizer_single_gpu.sh
```

If you don't want to use wavelet transformation, simply delete `${use_patcher}`, `${patch_size}` and `${patch_method}` arguments.
</details>

<details>
  <summary><b> 3. Train Text-to-Motion Model </b></summary>


First, please run the following command to inference all of the motion codes by the trained FSQ.
change the `${resume-pth}$` arguments to the path of tokenzier checkpoints of yourself.

```
bash scripts/train/train_t2m_get_codes.sh
```

Then, Train 3B model on multi-gpus by ZeRO-1 parallel, run the following command:

```
bash scripts/train/train_t2m_3B.sh
```

Train 7B model on multi-gpus by ZeRO-2 parallel, run the following command:

```
bash scripts/train/train_t2m_7B.sh
```

</details>

<details>
  <summary><b> 4. Evaluate the models </b></summary>

#### 4.1. Motion Reconstruction:

```
bash scripts/eval/eval_tokenizer.sh
```

#### 4.2. Text-to-Motion: 

```
bash scripts/eval/eval_t2m_3B.sh
bash scripts/eval/eval_t2m_7B.sh
```

</details>



## 🚨 Motion Postprocess

We provide a motion postprocess scripts to smooth and fix motion. Please execute the following command. A larger `${window_length}` will result in smoother motion.

```
cd postprocess/remove_sliding
bash scripts/run_remove_sliding.sh
```


</details>

## 🌹 Acknowledgement

We would like to thank the authors of the following repositories for their excellent work: 
[MotionLCM](https://github.com/ChenFengYe/motion-latent-diffusion), 
[T2M-GPT](https://github.com/Mael-zys/T2M-GPT), 
[MotionStreamer](https://github.com/zju3dv/MotionStreamer), 
[Scamo](https://github.com/shunlinlu/ScaMo_code),
[HumanML3D](https://github.com/EricGuo5513/HumanML3D).

## 📜 Citation

If you find this work useful, please consider citing our paper:

```bash
@inproceedings{motionmillion,
  title={Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data},
  author={},
  booktitle={Arxiv},
  pages={},
  year={2025}
}
```

## 📚 License

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

If you have any question, please contact at Ke Fan and cc to Shunlin Lu Chen and Jingbo Wang.
