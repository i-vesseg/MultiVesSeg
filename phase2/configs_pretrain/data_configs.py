from configs import transforms_config
from configs import transforms_config

#B

DATASETS = {
    'HQSWI': {
        'transforms': transforms_config.MyInpaintingTransforms,
        'train_source_root': "/home/galati/preprocessing/preprocess_MMs_B/train",
        'train_target_root': None,
        'val_source_root': "/home/galati/preprocessing/preprocess_MMs_B/val",
        'val_target_root': None,
        'test_source_root': "/home/galati/preprocessing/preprocess_MMs_B/test",
        'test_target_root': None,
    }
}

#OASIS

# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyInpaintingTransforms,
#         'train_source_root': "/home/galati/preprocessing/preprocessing_brain_data/preprocess_OASIS/train",
#         'train_target_root': None,
#         'val_source_root': "//home/galati/preprocessing/preprocessing_brain_data/preprocess_OASIS/val",
#         'val_target_root': None,
#         'test_source_root': "/home/galati/preprocessing/preprocessing_brain_data/preprocess_OASIS/test",
#         'test_target_root': None,
#     }
# }

#PDw

# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyInpaintingTransforms,
#         'train_source_root': "/home/galati/preprocessing/preprocess_SynthStrip_PDw/train",
#         'train_target_root': None,
#         'val_source_root': "//home/galati/preprocessing/preprocess_SynthStrip_PDw/val",
#         'val_target_root': None,
#         'test_source_root': "/home/galati/preprocessing/preprocess_SynthStrip_PDw/test",
#         'test_target_root': None,
#     }
# }