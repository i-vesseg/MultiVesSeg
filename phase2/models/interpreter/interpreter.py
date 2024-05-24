import os
import torch

from .classifier import pixel_classifier
import random
import gc

from PIL import Image
import numpy as np

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def latent_to_image(g_all, upsamplers, latents, return_upsampled_layers=False, use_style_latents=False,
                    process_out=True, return_stylegan_latent=False, dim=512,
                    return_only_im=False, noise=None, features_in=None, feature_scale=0):
    '''Given a input latent code, generate corresponding image and concatenated feature maps'''

    # assert (len(latents) == 1)  # for GPU memory constraints
    if not use_style_latents:
        # generate style_latents from latents
        style_latents = g_all.truncation(g_all.g_mapping(latents))
        style_latents = style_latents.clone()  # make different layers non-alias

    else:
        style_latents = latents

    if return_stylegan_latent:
        return  style_latents

    img_list, affine_layers = g_all.g_synthesis(style_latents, noise=noise, features_in=features_in, feature_scale=feature_scale)
    

    if return_only_im:
        if process_out:
            if img_list.shape[-2] > dim:
                img_list = upsamplers[-1](img_list)
            img_list = img_list.cpu().detach().numpy()
            img_list = process_image(img_list)
            img_list = np.transpose(img_list, (0, 2, 3, 1)).astype(np.uint8)
        return img_list, style_latents

    number_feautre = 0

    for item in affine_layers:
        number_feautre += item.shape[1]

    if return_upsampled_layers:
        affine_layers_upsamples = torch.FloatTensor(latents.shape[0], number_feautre, dim, dim).to(device)
        start_channel_index = 0
        for i in range(len(affine_layers)):
            len_channel = affine_layers[i].shape[1]
            affine_layers_upsamples[:, start_channel_index:start_channel_index + len_channel] = upsamplers[i](
                affine_layers[i])
            start_channel_index += len_channel
    else:
        affine_layers_upsamples = affine_layers

    if img_list.shape[-2] != dim:
        img_list = upsamplers[-1](img_list)

    if process_out:
        img_list = img_list.cpu().detach().numpy()
        img_list = process_image(img_list)
        img_list = np.transpose(img_list, (0, 2, 3, 1)).astype(np.uint8)

    return img_list, affine_layers_upsamples

class Interpreter(torch.nn.Module):
    def __init__(self, invert_intensity=False, label_nc=3):
        super(Interpreter, self).__init__()
        self.classifiers = torch.nn.ModuleDict({
            "source": pixel_classifier(numpy_class=label_nc, dim=64),
            "target": pixel_classifier(numpy_class=label_nc, dim=64),
        })
        self.invert_intensity = invert_intensity
        self.label_nc = label_nc

    def forward(self, latent, label, generator, dim, features_in=None, feature_scale=0, return_mask=True, ExSeg=False, is_semi=None):
        img, affine_layers = latent_to_image(
            generator, None, latent, dim=dim,
            return_upsampled_layers=False, use_style_latents=True,
            process_out=False, features_in=features_in, feature_scale=feature_scale,
        )
        if not return_mask:
            return [torch.mean(img, dim=1, keepdim=True)]

        img_seg_final = torch.zeros([len(img), self.label_nc, dim, dim], device=latent.device).type(img.dtype)
        is_target = label[:,1] > label[:,0]
        is_source = torch.logical_not(is_target)

        len_source = torch.sum(is_source)
        if len_source > 0:
            img_seg_final[is_source] = self.classifiers["source"](
                affine_layers, is_source, detach=not ExSeg, is_semi=is_semi
            ).reshape(len_source, dim, dim, -1).permute(0, 3, 1, 2)
        
        len_target = torch.sum(is_target)
        if len_target > 0:
            img_seg_final[is_target] = self.classifiers["target"]( 
                affine_layers, is_target, detach=True, is_semi=is_semi
            ).reshape(len_target, dim, dim, -1).permute(0, 3, 1, 2)
 
        if self.invert_intensity:
            img = torch.where(
                torch.argmax(img_seg_final, dim=1, keepdim=True) == 0, -1*img, img
            )
        
        return torch.mean(img, dim=1, keepdim=True), img_seg_final
