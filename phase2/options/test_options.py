from argparse import ArgumentParser
from configs.paths_config import model_paths


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str, help='Type of dataset/experiment to run')
        self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
        self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
        self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')

        self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')
        self.parser.add_argument(
            '--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.'
        )
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')
        self.parser.add_argument('--checkpoint_dir', default=None, type=str, help='Path to pSp model checkpoint')

        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--ssim_lambda', default=1.0, type=float, help='SSIM loss multiplier factor')
        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--ce_lambda', default=1.0, type=float, help='CE loss multiplier factor')
        self.parser.add_argument('--dice_lambda', default=1.0, type=float, help='Dice loss multiplier factor')

        self.parser.add_argument('--only_intra', action="store_true", help='Whether to perform inter-domain translation or not.')
        self.parser.add_argument('--disable_balanced_sampling', action="store_true", help='Disable balanced data sampling.')
        self.parser.add_argument('--disable_DSBN', action="store_true", help='Disable domain-specific batch normalization.')
        self.parser.add_argument('--disable_residuals', action="store_true", help='Disable residual connections between E and G.')
        self.parser.add_argument(
            '--invert_intensity', action="store_true", help='Invert intensity of target data. Unavailable at pretraining.'
        )
        self.parser.add_argument('--do_flip', action="store_true", help='Flip nose orientation from upward to downward.')   
        self.parser.add_argument(
            '--consider_all_vessels', action="store_true", help='Experts have labeled all vessels (within the weight mask).'
        )
        
        #federated
        self.parser.add_argument('--src_label', required=True, type=int, help='src domain label')
        self.parser.add_argument('--tgt_label', default=None, type=int, help='tgt domain label')
        self.parser.add_argument('--n_domains', default=3, type=int, help='Total number of domains')

    def parse(self):
        opts = self.parser.parse_args()
        opts.src_tgt_domains = [opts.src_label] + ([opts.tgt_label] if opts.tgt_label is not None else [])
        return opts
