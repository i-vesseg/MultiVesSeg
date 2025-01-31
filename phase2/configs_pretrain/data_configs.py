from configs import transforms_config
from configs import transforms_config

#OASIS (!!) SOURCE DOMAIN
DATASET_PATH ="PATH/To/PREPROCESSED/OASIS/"
DATASETS = {
    'HQSWI': {
        'transforms': transforms_config.MyTransforms,
        'train_source_root': f"{DATASET_PATH}/train",
        'train_target_root': None,
        'val_source_root': f"{DATASET_PATH}/val",
        'val_target_root': None,
        'test_source_root': f"{DATASET_PATH}/test",
        'test_target_root': None,
    }
}
