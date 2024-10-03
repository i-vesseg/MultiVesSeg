## MultiVesSeg &mdash; Official PyTorch Implementation
<div align="justify">

![Teaser image](./MRA_CTA_MRV.png)
**Picture:** *Maximum intensity projection (MIP) of a magnetic resonance angiography (left), MIP of a computed tomography angiography (center), and minimum intensity projection (mIP) of a magnetic resonance venography (right). All images are skull-stripped and viewed from the axial perspective.*

This repository contains the official PyTorch implementation of the following paper:

> **Multi-Domain Brain Vessel Segmentation Through Feature Disentanglement**<br>
> Francesco Galati, Rosa Cortese, Ferran Prados, Ninon Burgos, Maria A. Zuluaga<br>
> Under review
>
> **Abstract:** *The intricate morphology of brain vessels poses significant challenges for automatic segmentation models, which usually focus on a single imaging modality. However, accurately treating brain-related conditions requires a comprehensive understanding of the cerebrovascular tree regardless of the specific acquisition procedure. Through image-to-image translation, our framework effectively segments brain arteries and veins in various datasets, while avoiding domain-specific model design and data harmonization between the source and the target domain. This is accomplished by employing disentanglement techniques to independently manipulate different image properties, allowing to move from one domain to the other in a label-preserving manner. Specifically, we focus on the manipulation of vessel appearances during adaptation, while preserving spatial information such as shapes and locations, which are crucial for correct segmentation. Our evaluation demonstrates efficacy in bridging large and varied domain gaps across different medical centers, image modalities, and vessel types. Additionally, we conduct ablation studies on the optimal number of required annotations and other architectural choices. The results obtained highlight the robustness and versatility of our framework, demonstrating the potential of domain adaptation methodologies to perform cerebrovascular image segmentation accurately in multiple scenarios.*

</div>

## System requirements
- batchgenerators==0.24
- evalutils==0.4.2
- matplotlib==3.5.2
- MedPy==0.4.0
- nibabel==4.0.1
- nilearn==0.10.3
- opencv-python==4.6.0.66
- pytorch-msssim==0.2.1
- scikit-image==0.19.3
- SimpleITK==2.1.1.2
- tensorboard==2.9.1
- torch==1.9.1+cu111

## Preparing datasets for training
<div align="justify">

Folder `preprocessing` contains the Jupyter notebooks used to prepare the datasets utilized in our experiments: [OASIS](https://doi.org/10.1101/2019.12.13.19014902), [IXI](https://brain-development.org/ixi-dataset), and [TopCow](https://arxiv.org/abs/2312.17670). To preprocess your own dataset, create a new notebook by following the structure of the existing ones.

</div>

## Training networks

### Phase 1

To prepare the training data, you can run the following command:

```
python prepare_data.py --out ${DATA_dir} --size 512 --src_path ${SRC_dir}/train/ --tgt_path ${TGT_dir}/train/
```

Once the data is prepared, you can begin training the generator with the following command:

```
python -m torch.distributed.launch --nproc_per_node=2 train.py ${DATA_dir} --size 512 --n_sample 8 --iter 250000 --augment
```

### Phase 2
First, download the required pre-trained models and store them into a new directory:

| phase2/pretrained_models
| &boxvr;&nbsp; [alex.pth](https://github.com/richzhang/PerceptualSimilarity/raw/refs/heads/master/lpips/weights/v0.1/alex.pth)
| &boxvr;&nbsp; [alex_pretr.pth](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth)
| &boxvr;&nbsp; [backbone.pth](https://drive.google.com/file/d/1coFTz-Kkgvoc_gRT8JFzqCgeC3lAFWQp)

Before performing domain adaptation, you need to pre-train the source segmentation branch. To do so, follow these steps:

1) Edit the configuration files `data_configs.py` and `paths_config.py`  inside folder `configs_pretrain`.

2) Start the pre-training script.
```
python scripts/pretrain.py --exp_dir=${SRC_exp_dir} --batch_size 8 --start_from_latent_avg --label_nc=3 --max_steps=15000 --stylegan_weights ${PHASE1_dir}/checkpoint/final_checkpoint.pt --only_intra --src_label 0
```

Once the pre-training is complete, you can move to the target dataset:

1) Split the preprocessed target data into `labeled` and `unlabeled`.
```
mkdir ${TGT_dir}/train/labeled
mkdir ${TGT_dir}/train/unlabeled
mv ${TGT_dir}/train/*.npy ${TGT_dir}/train/unlabeled
mv ${TGT_dir}/train/unlabeled/${ID_1}_slice* ${TGT_dir}/train/labeled
mv ${TGT_dir}/train/unlabeled/${ID_2}_slice* ${TGT_dir}/train/labeled
mv ${TGT_dir}/train/unlabeled/${ID_3}_slice* ${TGT_dir}/train/labeled
```

2) Edit the configuration files `data_configs.py` and `paths_config.py`  inside folder `configs_train`.

3) Start the training script.
```
python scripts/train.py --exp_dir=${TGT_exp_dir} --start_from_latent_avg --label_nc=3 --max_steps=20000 --checkpoint_dir=${SRC_exp_dir}/checkpoints  --one_target_slice --src_label 0 --tgt_label 1
```

## Running inference

To start the inference script, you can use this command:
```
python scripts/inference.py --metadata=${INFO_path} --exp_dir=${INF_dir} --start_from_latent_avg --label_nc=4 --checkpoint_dir=${TGT_exp_dir}/checkpoints --src_label 0 --tgt_label 1
```

Please note that the .pkl file located at ${INFO_path} was generated by the Jupyter notebook during preprocessing, provided it was executed correctly.
