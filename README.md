# MSSC-Former

This repository contains the supported pytorch code and configuration files to reproduce of MSSC-Former.

![MSSC-Former](img/Architecture_overview.jpg?raw=true)

Parts of codes are borrowed from [nn-UNet](https://github.com/MIC-DKFZ/nnUNet). For detailed configuration of the dataset, please refer to [nn-UNet](https://github.com/MIC-DKFZ/nnUNet).

## Environment

Please prepare an environment with Python 3.7, Pytorch 1.7.1, and Windows 10.

## Dataset Preparation

Datasets can be acquired via following links:

- [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
- [The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
- [Brain_tumor](http://medicaldecathlon.com/)
- [Heart](http://medicaldecathlon.com/)

## Dataset Set

After you have downloaded the datasets, you can follow the settings in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for path configurations and preprocessing procedures. Finally, your folders should be organized as follows:

```
./MSSCFormer/
./DATASET/
  ├── MSSCFormer_raw/
      ├── MSSCFormer_raw_data/
          ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task03_tumor/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task04_Heart/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
      ├── MSSCFormer_cropped_data/
  ├── MSSCFormer_trained_models/
  ├── MSSCFormer_preprocessed/
```

## Preprocess Data

- MSSCFormer_convert_decathlon_task -i D:\Codes\Medical_image\UploadGitHub\MSSCFormer\DATASET\MSSCFormer_raw\MSSCFormer_raw_data
- MSSCFormer_plan_and_preprocess -t 2

## Functions of scripts

- **Network architecture:**
  - MSSCFormer\MSSCFormer\network_architecture\MSSCFormer_acdc.py``
  - MSSCFormer\MSSCFormer\network_architecture\MSSCFormer_synapse.py``
  - MSSCFormer\MSSCFormer\network_architecture\MSSCFormer_tumor.py``
  - MSSCFormer\MSSCFormer\network_architecture\MSSCFormer_heart.py``
- **Trainer for dataset:**
  - MSSCFormer\MSSCFormer\training\network_training\MSSCFormerTrainerV2_MSSCFormer_acdc.py``
  - MSSCFormer\MSSCFormer\training\network_training\MSSCFormerTrainerV2_MSSCFormer_synapse.py``
  - MSSCFormer\MSSCFormer\training\network_training\MSSCFormerTrainerV2_MSSCFormer_tumor.py``
  - MSSCFormer\MSSCFormer\training\network_training\MSSCFormerTrainerV2_MSSCFormer_heart.py``

## Train Model

- python run_training.py  3d_fullres  MSSCFormerTrainerV2_MSSCFormer_synapse 2 0


## Test Model

- python predict.py -i D:\Codes\Medical_image\UploadGitHub\MSSCFormer\DATASET\MSSCFormer_raw\MSSCFormer_raw_data\Task002_Synapse\imagesTs
  -o D:\Codes\Medical_image\UploadGitHub\MSSCFormer\DATASET\MSSCFormer_raw\MSSCFormer_raw_data\Task002_Synapse\imagesTs_infer
  -m D:\Codes\Medical_image\UploadGitHub\MSSCFormer\DATASET\MSSCFormer_trained_models\MSSCFormer\3d_fullres\Task002_Synapse\MSSCFormerTrainerV2_MSSCFormer_synapse__MSSCFormerPlansv2.1
  -f 0

- python MSSCFormer/inference_synapse.py

## Acknowledgements

This repository makes liberal use of code from:

- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) 
- [nnFormer](https://github.com/282857341/nnFormer)
- [UNETR++](https://github.com/Amshaker/unetr_plus_plus)
