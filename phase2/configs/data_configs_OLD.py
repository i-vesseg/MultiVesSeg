from configs import transforms_config
from configs import transforms_config

#HQSWI


# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyTransforms,
#         'train_source_root': "/data/falcetta/A2V_experiments/OASIS_preprocessed/preprocess_OASIS/train",
#         'train_target_root': {
#             "unlabeled": "/data/falcetta/A2V_experiments/IXI_preprocessed/preprocess_IXI/train/unlabeled",
#             "labeled": "/data/falcetta/A2V_experiments/IXI_preprocessed/preprocess_IXI/train/labeled",
#         },
#         'val_source_root': None,
#         'val_target_root': {
#             "labeled": "/data/falcetta/A2V_experiments/IXI_preprocessed/preprocess_IXI/val",
#         },
#         'test_source_root': None,
#         'test_target_root': {
#             "labeled": "/data/falcetta/A2V_experiments/IXI_preprocessed/preprocess_IXI/test",
#         },
#     }
# }

DATASETS = {
    'HQSWI': {
        'transforms': transforms_config.MyTransforms,
        'train_source_root': "/data/falcetta/A2V_experiments/OASIS_preprocessed/preprocess_OASIS/train",
        'train_target_root': {
            #"unlabeled": "/data/falcetta/A2V_experiments/TOPCOW_preprocessed/preprocess_TopCow_all/train/unlabeled",
            #"labeled": "/data/falcetta/A2V_experiments/TOPCOW_preprocessed/preprocess_TopCow_all/train/labeled",
            "unlabeled": "/data/falcetta/A2V_experiments/GRENOBLE_CTs_processed/preprocess_GrenobleCTs/train/unlabeled",
            "labeled": "/data/falcetta/A2V_experiments/GRENOBLE_CTs_processed/preprocess_GrenobleCTs/train/labeled",
        },
        'val_source_root': None,
        'val_target_root': {
            "labeled": "/data/falcetta/A2V_experiments/GRENOBLE_CTs_processed/preprocess_GrenobleCTs/val",
        },
        'test_source_root': None,
        'test_target_root': {
            #"labeled": "/data/falcetta/A2V_experiments/TOPCOW_preprocessed/preprocess_TopCow_all/test",
            #"labeled": "/data/falcetta/TOPCOW_RES/A2V/23/CT/preprocess_TOPCOW_23CT_TEST/test",
            #"labeled": "/data/falcetta/TOPCOW_RES/A2V/23/MR/preprocess_TOPCOW_23MR_TEST/test",
            #"labeled": "/data/galati/brain_data/ITKTubeTK/fold1/A2VpreproccTs/preprocess_ITKTubeTK_A2V/test",
            #"labeled": "/data/falcetta/LONDON_TEST/OUT/preprocess_LONDON_TEST/test"
            "labeled": "/data/falcetta/A2V_experiments/GRENOBLE_CTs_processed/preprocess_GrenobleCTs/test"
        },
    }
}

#IXI

# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyTransforms,
#         'train_source_root': "/data/galati/brain_data/preprocessing_brain_data/preprocess_OASIS/train",
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

#TopCow

# DATASETS = {
#     'HQSWI': {
#         'transforms': transforms_config.MyTransforms,
#         'train_source_root': "/data/galati/brain_data/preprocessing_brain_data/preprocess_OASIS/train",
#         'train_target_root': {
#             "unlabeled": "/data/galati/brain_data/preprocessing_brain_data/preprocess_TopCow/train/unlabeled",
#             "labeled": "/data/galati/brain_data/preprocessing_brain_data/preprocess_TopCow/train/labeled",
#         },
#         'val_source_root': None,
#         'val_target_root': {
#             "labeled": "/data/galati/brain_data/preprocessing_brain_data/preprocess_TopCow/val",
#         },
#         'test_source_root': None,
#         'test_target_root': {
#             "labeled": "/data/galati/brain_data/preprocessing_brain_data/preprocess_TopCow/test",
#         },
#     }
# }
