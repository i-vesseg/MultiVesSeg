from configs import transforms_config

DATASETS = {
    'HQSWI': {
        'transforms': transforms_config.MyTransforms,
        'train_source_root': "",
        'train_target_root': {
            "unlabeled": "",
            "labeled": "",
        },
        'val_source_root': None,
        'val_target_root': {
        },
        'test_source_root': None,
        'test_target_root': {
            "labeled": f"PATH/THAT/I/WANT/test"
        },
    }
}
