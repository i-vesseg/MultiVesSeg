"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
from models.encoders import feature_style_encoder
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
from models.interpreter.interpreter import Interpreter
import gc

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.set_opts(opts)
        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = feature_style_encoder.fs_encoder_v2(
            self.opts.n_styles, model_paths['arcface_model'],
            residual=not self.opts.disable_residuals, DSBN=not self.opts.disable_DSBN
        )
        self.decoder = Generator(self.opts.output_size, 512, 8, opts.n_domains)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.interpreter = Interpreter(opts.invert_intensity, label_nc=opts.label_nc)

        #inference
        print("Reducing batch norm for problems")
        for m in self.modules():
            for child in m.children():
                if type(child) == nn.BatchNorm2d or type(child) == nn.BatchNorm1d:
                    child.track_running_stats = False
                    child.running_mean = None
                    child.running_var = None

        # Load weights if needed
        self.load_weights()
        
        for m in self.modules():
            for child in m.children():
                if type(child) == nn.BatchNorm2d or type(child) == nn.BatchNorm1d:
                    child.track_running_stats = False
                    child.running_mean = None
                    child.running_var = None

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            if len(self.opts.src_tgt_domains) == len(ckpt['latent_avg']):
                self.encoder.load_state_dict(get_keys(ckpt, 'module.encoder'), strict=True)
            else:
                print("Encoder is not being loaded")
            self.decoder.load_state_dict(get_keys(ckpt, 'module.decoder'), strict=False)
            self.interpreter.load_state_dict(get_keys(ckpt, 'module.interpreter'), strict=True)
            if not self.opts.only_intra:
                requires_grad(self.interpreter.classifiers["source"], False)
            self.__load_latent_avg(ckpt)
        else:
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
            if self.opts.learn_in_w:
                self.__load_latent_avg(ckpt, repeat=1)
            else:
                self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

    def forward(self, x, label, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, feature_scale=1, return_mask=True, ExSeg=False, is_semi=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x, label)
            if not self.opts.disable_residuals:
                codes, residuals = codes
            if self.opts.start_from_latent_avg:
                if self.opts.learn_in_w:
                    codes = codes + self.latent_avg[torch.argmax(label, dim=-1)].type(codes.dtype)
                else:
                    codes = codes + self.latent_avg[torch.argmax(label, dim=-1)][:, None].type(codes.dtype)

            if not self.opts.disable_residuals:
                residuals = self.encoder.convertContent(residuals[:-1][::-1], residuals[-1])
            else:
                residuals = None

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images = torch.cat([*self.interpreter(
            codes,
            label,
            self.decoder,
            self.opts.output_size,
            features_in=residuals,
            feature_scale=feature_scale,
            return_mask=return_mask,
            ExSeg=ExSeg,
            is_semi=is_semi,
        )], dim=1)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, codes
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        with torch.no_grad():
            if 'latent_avg' in ckpt and len(self.opts.src_tgt_domains) == len(ckpt['latent_avg']):
                print("Loading latent_avg from checkpoint")
                self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
                if repeat is not None:
                    self.latent_avg = self.latent_avg.repeat(repeat, 1)
                self.latent_avg = torch.nn.Parameter(self.latent_avg)
            else:
                self.latent_avg = None
