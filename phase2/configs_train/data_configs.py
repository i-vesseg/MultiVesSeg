from configs import transforms_config
from configs import transforms_config

#HQSWI

DATASETS = {
    'HQSWI': {
        'transforms': transforms_config.MyInpaintingTransforms,
        'train_source_root': "/home/galati/A2V_FL/phase2/fake_OASIS",
        'train_target_root': {
            "unlabeled": "/data/galati/brain_data/preprocessing_brain_data/preprocess_HQSWI/train/unlabeled",
            "labeled": "/data/galati/brain_data/preprocessing_brain_data/preprocess_HQSWI/train/labeled",
        },
        'val_source_root': None,
        'val_target_root': {
            "labeled": "/data/galati/brain_data/preprocessing_brain_data/preprocess_HQSWI/val",
        },
        'test_source_root': None,
        'test_target_root': {
            "labeled": "/data/galati/brain_data/preprocessing_brain_data/preprocess_HQSWI/test",
        },
    }
}

#OCTA500

# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyInpaintingTransforms,
#         'train_source_root': "/home/galati/A2V_FL/phase2/fake_OASIS",
#         'train_target_root': {
#             "unlabeled": "/home/galati/preprocessing/preprocess_OCTA500/train/unlabeled",
#             "labeled": "/home/galati/preprocessing/preprocess_OCTA500/train/labeled",
#         },
#         'val_source_root': None,
#         'val_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_OCTA500/val",
#         },
#         'test_source_root': None,
#         'test_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_OCTA500/test",
#         },
#     }
# }

#SynthStrip_PET

# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyInpaintingTransforms,
#         'train_source_root': "/home/galati/A2V_FL/phase2/fake_SynthStrip_PDw",
#         'train_target_root': {
#             "unlabeled": "/home/galati/preprocessing/preprocess_SynthStrip_PET/train/unlabeled",
#             "labeled": "/home/galati/preprocessing/preprocess_SynthStrip_PET/train/labeled",
#         },
#         'val_source_root': None,
#         'val_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_SynthStrip_PET/val",
#         },
#         'test_source_root': None,
#         'test_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_SynthStrip_PET/test",
#         },
#     }
# }

#SynthStrip_CT

# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyInpaintingTransforms,
#         'train_source_root': "/home/galati/A2V_FL/phase2/fake_SynthStrip_PDw",
#         'train_target_root': {
#             "unlabeled": "/home/galati/preprocessing/preprocess_SynthStrip_CT/train/unlabeled",
#             "labeled": "/home/galati/preprocessing/preprocess_SynthStrip_CT/train/labeled",
#         },
#         'val_source_root': None,
#         'val_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_SynthStrip_CT/val",
#         },
#         'test_source_root': None,
#         'test_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_SynthStrip_CT/test",
#         },
#     }
# }

#SynthStrip_T2w

# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyInpaintingTransforms,
#         'train_source_root': "/home/galati/A2V_FL/phase2/fake_SynthStrip_PDw",
#         'train_target_root': {
#             "unlabeled": "/home/galati/preprocessing/preprocess_SynthStrip_T2w/train/unlabeled",
#             "labeled": "/home/galati/preprocessing/preprocess_SynthStrip_T2w/train/labeled",
#         },
#         'val_source_root': None,
#         'val_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_SynthStrip_T2w/val",
#         },
#         'test_source_root': None,
#         'test_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_SynthStrip_T2w/test",
#         },
#     }
# }

#IXI

# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyInpaintingTransforms,
#         'train_source_root': "/home/galati/A2V_FL/phase2/fake_OASIS",
#         'train_target_root': {
#             "unlabeled": "/data/galati/brain_data/preprocessing_brain_data/preprocess_IXI/train/unlabeled",
#             "labeled": "/data/galati/brain_data/preprocessing_brain_data/preprocess_IXI/train/labeled",
#         },
#         'val_source_root': None,
#         'val_target_root': {
#             "labeled": "/data/galati/brain_data/preprocessing_brain_data/preprocess_IXI/val",
#         },
#         'test_source_root': None,
#         'test_target_root': {
#             "labeled": "/data/galati/brain_data/preprocessing_brain_data/preprocess_IXI/test",
#         },
#     }
# }

#MMs_A

# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyInpaintingTransforms,
#         'train_source_root': "/home/galati/A2V_FL/phase2/fake_MMs_B",
#         'train_target_root': {
#             "unlabeled": "/home/galati/preprocessing/preprocess_MMs_A/train/unlabeled",
#             "labeled": "/home/galati/preprocessing/preprocess_MMs_A/train/labeled",
#         },
#         'val_source_root': None,
#         'val_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_MMs_A/val",
#         },
#         'test_source_root': None,
#         'test_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_MMs_A/test",
#         },
#     }
# }

#MMs_D

# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyInpaintingTransforms,
#         'train_source_root': "/home/galati/A2V_FL/phase2/fake_MMs_B",
#         'train_target_root': {
#             "unlabeled": "/home/galati/preprocessing/preprocess_MMs_D_aligned/train/unlabeled",
#             "labeled": "/home/galati/preprocessing/preprocess_MMs_D_aligned/train/labeled",
#         },
#         'val_source_root': None,
#         'val_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_MMs_D_aligned/val",
#         },
#         'test_source_root': None,
#         'test_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_MMs_D_aligned/test",
#         },
#     }
# }

#MMs_C

# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyInpaintingTransforms,
#         'train_source_root': "/home/galati/A2V_FL/phase2/fake_MMs_B",
#         'train_target_root': {
#             "unlabeled": "/home/galati/preprocessing/preprocess_MMs_C/train/unlabeled",
#             "labeled": "/home/galati/preprocessing/preprocess_MMs_C/train/labeled",
#         },
#         'val_source_root': None,
#         'val_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_MMs_C/val",
#         },
#         'test_source_root': None,
#         'test_target_root': {
#             "labeled": "/home/galati/preprocessing/preprocess_MMs_C/test",
#         },
#     }
# }
