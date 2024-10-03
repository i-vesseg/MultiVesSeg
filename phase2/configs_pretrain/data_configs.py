from configs import transforms_config
from configs import transforms_config

#OASIS

DATASETS = {
    'HQSWI': {
        'transforms': transforms_config.MyTransforms,
        'train_source_root': "/home/galati/preprocessing/preprocessing_brain_data/preprocess_OASIS/train",
        'train_target_root': None,
        'val_source_root': "/home/galati/preprocessing/preprocessing_brain_data/preprocess_OASIS/val",
        'val_target_root': None,
        'test_source_root': "/home/galati/preprocessing/preprocessing_brain_data/preprocess_OASIS/test",
        'test_target_root': None,
    }
}
