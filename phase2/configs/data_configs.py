from configs import transforms_config
from configs import transforms_config

#HQSWI


DATASETS = {
    'HQSWI': { # Do NOT change this key
        'transforms': transforms_config.MyTransforms,
        'train_source_root': "/data/falcetta/A2V_experiments/OASIS_preprocessed/preprocess_OASIS/train",
        'train_target_root': {
            "unlabeled": "/home/geninana/data_ssd/DqnieleF/A2V_experiments/TOF_TEST_ADAPT_preprocessed/preprocess_TOF_TEST_ADAPT/train/unlabeled", #TOF_GRENOBLE
            "labeled": "/home/geninana/data_ssd/DqnieleF/A2V_experiments/TOF_TEST_ADAPT_preprocessed/preprocess_TOF_TEST_ADAPT/train/labeled",
            # "unlabeled": "/home/geninana/data_ssd/DqnieleF/A2V_experiments/CT_TAS_GRENOBLE_TEST_preprocessed/preprocess_CT_TAS_GRENOBLE_TEST/train/unlabeled", #$CT_TAS_GRENOBLE
            # "labeled": "/home/geninana/data_ssd/DqnieleF/A2V_experiments/CT_TAS_GRENOBLE_TEST_preprocessed/preprocess_CT_TAS_GRENOBLE_TEST/train/labeled",
        },
        'val_source_root': None,
        'val_target_root': {
            "labeled": "/home/geninana/data_ssd/DqnieleF/A2V_experiments/TOF_TEST_ADAPT_preprocessed/preprocess_TOF_TEST_ADAPT/val", #TOF_GRENOBLE
            # labeled": "/home/geninana/data_ssd/DqnieleF/A2V_experiments/CT_TAS_GRENOBLE_TEST_preprocessed/preprocess_CT_TAS_GRENOBLE_TEST/val" #$CT_TAS_GRENOBLE
        },
        'test_source_root': None,
        'test_target_root': {
            #"labeled": "/home/geninana/data_ssd/DqnieleF/A2V_experiments/TOF_TEST_ADAPT_preprocessed/preprocess_TOF_TEST_ADAPT/test", #TOF_GRENOBLE
            #"labeled": "/home/geninana/data_ssd/DqnieleF/A2V_experiments/CT_TAS_GRENOBLE_TEST_preprocessed/preprocess_CT_TAS_GRENOBLE_TEST/test" #$CT_TAS_GRENOBLE
            "labeled": "/home/geninana/data_ssd/DqnieleF/A2V_experiments/CT_GRENOBLE_5_preprocessed/preprocess_CT_GRENOBLE_5/test"
        },
    }
}
