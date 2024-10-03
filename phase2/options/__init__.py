from argparse import ArgumentParser
from options.singleton import Singleton

class Options(metaclass=Singleton):
    def __init__(self, is_train):
        self.is_tran = is_train
        self.parser = ArgumentParser()
        self.initialize()
        if is_train:
            self.initialize_train()
        else:
            self.initialize_inference()

    def initialize(self):
        #paths
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_dir', default=None, type=str, help='Path to pSp model checkpoint')
        self.parser.add_argument(
            '--arcface_model', default="./pretrained_models/backbone.pth", type=str, help='Path to pretrained arcface model'
        )
        self.parser.add_argument(
            '--alex_model', default="./pretrained_models/alex.pth", type=str, help='Path #1 to pretrained alex model'
        )
        self.parser.add_argument(
            '--alex_pretr_model', default="./pretrained_models/alex_pretr.pth", type=str, help='Path #2 to pretrained alex model'
        )
        
        #dataset
        self.parser.add_argument('--dataset_type', default='HQSWI', type=str, help='Type of dataset/experiment to run')
        self.parser.add_argument('--input_nc', default=1, type=int, help='Number of input image channels to the psp encoder')
        self.parser.add_argument('--label_nc', default=3, type=int, help='Number of input label channels to the psp encoder')
        self.parser.add_argument('--output_size', default=512, type=int, help='Output size of generator')        #federated
        self.parser.add_argument('--src_label', required=True, type=int, help='Src domain label')
        self.parser.add_argument('--tgt_label', default=None, type=int, help='Tgt domain label')
        self.parser.add_argument('--n_domains', default=3, type=int, help='Total number of domains represented in W (from phase1)')

        #data loader
        self.parser.add_argument('--test_batch_size', default=8, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=6, type=int, help='Number of test/inference dataloader workers')

        #latent space
        self.parser.add_argument(
            '--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.'
        )
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')

        #losses
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--ssim_lambda', default=1.0, type=float, help='SSIM loss multiplier factor')
        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--ce_lambda', default=1.0, type=float, help='CE loss multiplier factor')
        self.parser.add_argument('--ce_weights', nargs='+', type=float, help='CE loss class weights')
        self.parser.add_argument('--dice_lambda', default=1.0, type=float, help='Dice loss multiplier factor')
        self.parser.add_argument('--dice_weights', nargs='+', type=float, help='Dice loss class weights')

        #extra features
        self.parser.add_argument('--only_intra', action="store_true", help='Whether to perform inter-domain translation or not.')
        self.parser.add_argument('--disable_residuals', action="store_true", help='Disable residual connections between E and G.')
        self.parser.add_argument('--disable_DSBN', action="store_true", help='Disable domain-specific batch normalization.')
        self.parser.add_argument(
            '--invert_intensity', action="store_true", help='Invert intensity of target data.'
        )
        self.parser.add_argument(
            '--consider_only_vessels_within_brain', action="store_true",
            help='By default, experts have labeled all vessels (within the weight mask). If activated: despite weight masks are not provided, only vessels within the brain have been labeled, not all the visible ones -> the model is not penalized if it annotates vessels outside the brain.'
        )
        
    def initialize_train(self):
        #paths
        self.parser.add_argument('--stylegan_weights', default=None, type=str, help='Path to StyleGAN model weights')
        
        #training parameters
        self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
        self.parser.add_argument('--workers', default=6, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        
        #logging
        self.parser.add_argument('--image_interval', default=500, type=int, help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=100, type=int, help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')
        
        #extra features
        self.parser.add_argument('--train_decoder', action='store_true', help='Whether to train the decoder model')
        self.parser.add_argument('--one_target_slice', action="store_true", help='Use only one annotated slice at target.')
        self.parser.add_argument('--disable_balanced_sampling', action="store_true", help='Disable balanced data sampling.')
        self.parser.add_argument('--use_da', action="store_true", help='Use data augmentation.')
    
    def initialize_inference(self):
        #paths
        self.parser.add_argument('--metadata', required=True, type=str, help='Path to metadata from preprocessing')
        
        #postprocessing
        self.parser.add_argument('--do_flip', action="store_true", help='Flip nose orientation from upward to downward.')
        
        #extra features
        self.parser.add_argument('--ensemble_size', default=5, type=int, help='Number of checkpoints to use at inference.')

    def parse(self):
        self.opts = self.parser.parse_args()
        self.opts.src_tgt_domains = [self.opts.src_label] + ([self.opts.tgt_label] if self.opts.tgt_label is not None else [])
        
        assert self.opts.input_nc == 1, "1 is hard-coded in the code, as the project is intended to run with medical images (easy fix)"
        assert not (self.opts.only_intra and self.opts.invert_intensity), "Invert intensity is unavailable at pretraining."
        assert self.opts.ce_weights is None or len(self.opts.ce_weights) == self.opts.label_nc, "Wrong number of CE weights"
        assert self.opts.dice_weights is None or len(self.opts.dice_weights) == self.opts.label_nc, "Wrong number of dice weights"
        
        return self.opts
