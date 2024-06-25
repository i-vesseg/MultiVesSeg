import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import ce_loss, dice_loss, ssim_loss
from criteria.lpips.lpips import LPIPS
from configs import data_configs
from datasets.images_dataset import ImagesDataset, MyRandomSampler
from models.psp import pSp

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

import gc
import random
import numpy as np
from medpy.metric import binary
import nibabel as nib
import copy
import pickle
from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize

def DC(prediction, target):
    try: return binary.dc(prediction, target)
    except Exception: return 0

def HD(prediction, target):
    try: return binary.hd(prediction, target)
    except Exception: return np.inf

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(0)

#torch.autograd.set_detect_anomaly(True)

class progress_count():
    def __init__(self):
        self.counter = -1
    def __call__(self):
        self.counter += 1
        self.counter %= 8
        if self.counter == 4:
            self.counter += 1
        return self.counter % 4
progress_count = progress_count()

class Coach:
    def __init__(self, opts, global_step):
        self.opts = opts

        self.global_step = global_step

        self.device = 'cuda'
        self.opts.device = self.device

        # Initialize network
        self.net = pSp(self.opts)
        self.net = torch.nn.DataParallel(self.net, device_ids=[0]).to(self.device)#PARALLEL

        assert self.net.module.latent_avg is not None

        # Initialize loss
        if self.opts.l2_lambda > 0:
            self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.ssim_lambda > 0:
            self.ssim_loss = ssim_loss.SSIM(data_range=1, size_average=True, channel=1).to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.dice_lambda > 0:
            self.dice_loss = dice_loss.DiceLoss().to(self.device).eval()
        if self.opts.ce_lambda > 0:
            self.ce_loss = ce_loss.CELoss().to(self.device).eval()

        # Initialize dataset
        self.test_dataset = self.configure_datasets()
        self.test_dataloader = DataLoader(self.test_dataset,
                                        batch_size=self.opts.test_batch_size,
                                        shuffle=False,
                                        num_workers=int(self.opts.test_workers),
                                        drop_last=False)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = [(f"best_model_{i}.pth", 0) for i in range(5)]

    def infer(self):
        self.net.eval()
        agg_loss_dict = []
        original = None
        predicted_inter = None
        predicted_intra = None
        curr_img_name = None
        translation = None
        metrics = {}
        pred_dict = {}
        
        for batch_idx, batch in enumerate(self.test_dataloader):
            x, y, w, labels, _ = batch
            all_paths = self.test_dataloader.dataset.paths_tof + self.test_dataloader.dataset.paths_swi
            img_names = all_paths[
                batch_idx * self.opts.test_batch_size : batch_idx * self.opts.test_batch_size + len(x)
            ]
            img_names = [os.path.basename(img_name).split("_slice")[0] for img_name in img_names]
            if curr_img_name is None:
                curr_img_name = img_names[0]

            with torch.no_grad():
                x, y, w, labels = (
                    x.to(self.device).float(), y.to(self.device).float(), w.to(self.device).float(), labels.to(self.device).float()
                )

                is_swi = labels[:,1] > labels[:,0]
                is_tof = torch.logical_not(is_swi)
                
                #INTRA DOMAIN BRANCH
                y_hat, latent = self.net(x, labels, resize=False, return_latents=True, feature_scale=min(1.0, 0.0001*self.global_step))
                
                slice_weight = w.clone()
                slice_prediction = y_hat[:,1:].clone()
                slice_original = y.clone()
                w = self.net.module.face_pool(w)
                y_hat = self.net.module.face_pool(y_hat)
                y = self.net.module.face_pool(y)
                
                loss_intra, cur_loss_dict_intra, id_logs_intra = self.calc_loss(x, y, y_hat, weight_msk_loss=w)

                original = np.concatenate([
                    original, slice_original.cpu().numpy()
                ]) if original is not None else slice_original.cpu().numpy()
                predicted_intra = np.concatenate([
                    predicted_intra, slice_prediction.cpu().numpy()
                ]) if predicted_intra is not None else slice_prediction.cpu().numpy()
                
                loss = loss_intra
                cur_loss_dict = {**cur_loss_dict_intra}
                assert id_logs_intra is None, f"{id_logs_intra}"

                #INTER DOMAIN BRANCH
                if not self.opts.only_intra:
                    y_hat = self.net(
                        x, torch.logical_not(labels).type(labels.dtype),
                        resize=False, return_latents=False, feature_scale=min(1.0, 0.0001*self.global_step)
                    )
                    
                    slice_prediction = y_hat[:,1:].clone()
                    slice_translation = y_hat[:,:1].clone()
                    y_hat = self.net.module.face_pool(y_hat)
                    
                    y_cycle = y_hat[:, :1].clone()
                    y_cycle, latent = self.net(
                        y_cycle, labels, return_latents=True, feature_scale=min(1.0, 0.0001*self.global_step)
                    )
                    
                    loss_inter, cur_loss_dict_inter, id_logs_inter = self.calc_loss(
                        x, y, torch.cat([y_cycle[:,:1], y_hat[:,1:]], dim=1), weight_msk_loss=w
                    )

                    predicted_inter = np.concatenate([
                        predicted_inter, slice_prediction.cpu().numpy()
                    ]) if predicted_inter is not None else slice_prediction.cpu().numpy()
                    if self.opts.invert_intensity:
                        slice_translation = torch.where(
                            torch.argmax(slice_prediction, dim=1, keepdim=True) == 0, -1*slice_translation, slice_translation
                        )
                    translation = np.concatenate([
                        translation, slice_translation.cpu().numpy()
                    ]) if translation is not None else slice_translation.cpu().numpy()
                    
                    loss += loss_inter
                    cur_loss_dict = {
                        **cur_loss_dict,
                        **{"inter_" + k: v for k,v in cur_loss_dict_inter.items()}
                    }
                    assert id_logs_inter is None, f"{id_logs_inter}"

            agg_loss_dict.append(cur_loss_dict)

            if curr_img_name != img_names[-1] or batch_idx == len(self.test_dataloader) - 1:
                if curr_img_name not in img_names:
                    until = -1 * len(x)
                elif curr_img_name in img_names:
                    until = img_names[::-1].index(curr_img_name) * -1
                if until == 0:
                    until = None

                volume_orig = original[:until]
                volume_intra = predicted_intra[:until]
                if not self.opts.only_intra:
                    volume_inter = predicted_inter[:until]
                    volume_trans = translation[:until][:,0]
                    volume_ensemble = np.average([predicted_inter[:until], predicted_intra[:until]], axis=0, weights=[0.5, 0.5])
                
                pred_dict[curr_img_name] = {
                    "intra":  volume_intra,
                    "inter":  volume_inter if not self.opts.only_intra else 0,
                    "trans":  volume_trans if not self.opts.only_intra else 0,
                }
                
                volume_orig = np.argmax(volume_orig, axis=1)
                volume_intra = np.argmax(volume_intra, axis=1)
                if not self.opts.only_intra:
                    volume_inter = np.argmax(volume_inter, axis=1)
                    volume_ensemble = np.argmax(volume_ensemble, axis=1)
                
                metrics[curr_img_name] = {}
                for label_idx in range(self.opts.label_nc):
                    if label_idx == 2 and self.opts.consider_only_vessels_within_brain:
                        assert label_idx == len(self.opts.label_nc)-1, "This flag was intended for vessels, when brain=1 and vessels=2"
                        roi = np.logical_not(np.isclose(volume_orig, 0))#experts have labeled only vessels inside the brain
                    else:
                        roi = np.full(volume_orig.shape, True, dtype=bool)#experts have labeled all (within the weight mask)
                    binary_orig = np.logical_and(np.isclose(volume_orig, label_idx), roi)
                    binary_intra = np.logical_and(np.isclose(volume_intra, label_idx), roi)
                    if not self.opts.only_intra:
                        binary_inter = np.logical_and(np.isclose(volume_inter, label_idx), roi)
                        binary_ensemble = np.logical_and(np.isclose(volume_ensemble, label_idx), roi)
                    
                    metrics[curr_img_name][f"intra_dice_{label_idx}"] = DC(binary_intra, binary_orig)
                    metrics[curr_img_name][f"intra_hausdorff_{label_idx}"] = HD(binary_intra, binary_orig)
                    if not self.opts.only_intra:
                        metrics[curr_img_name][f"inter_dice_{label_idx}"] = DC(binary_inter, binary_orig)
                        metrics[curr_img_name][f"inter_hausdorff_{label_idx}"] = HD(binary_inter, binary_orig)
                        metrics[curr_img_name][f"dice_{label_idx}"] = DC(binary_ensemble, binary_orig)
                        metrics[curr_img_name][f"hausdorff_{label_idx}"] = HD(binary_ensemble, binary_orig)
                
                metrics[curr_img_name][f"intra_dice_foreground"] = DC(volume_intra, volume_orig)
                metrics[curr_img_name][f"intra_hausdorff_foreground"] = HD(volume_intra, volume_orig)
                if not self.opts.only_intra:
                    metrics[curr_img_name][f"inter_dice_foreground"] = DC(volume_inter, volume_orig)
                    metrics[curr_img_name][f"inter_hausdorff_foreground"] = HD(volume_inter, volume_orig)
                    metrics[curr_img_name][f"dice_foreground"] = DC(volume_ensemble, volume_orig)
                    metrics[curr_img_name][f"hausdorff_foreground"] = HD(volume_ensemble, volume_orig)
                
                print(curr_img_name, metrics[curr_img_name])

                original = original[until:] if until is not None else None
                predicted_intra = predicted_intra[until:] if until is not None else None
                if not self.opts.only_intra:
                    predicted_inter = predicted_inter[until:] if until is not None else None
                    translation = translation[until:] if until is not None else None
                
                yield curr_img_name, pred_dict[curr_img_name]
                del pred_dict[curr_img_name]
                curr_img_name = img_names[-1] if until is not None else None

        metrics = {k: np.mean([metrics[patient_id][k] for patient_id in metrics]) for k in metrics[list(metrics.keys())[0]]}
        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        loss_dict = {**metrics, **loss_dict}
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')
        
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best, best_path='best_model.pt'):
        save_name = best_path if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(f'**Best saved at {best_path}**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        print(f'Loading dataset for {self.opts.dataset_type}')
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        test_dataset = ImagesDataset(
            opts=self.opts,
            source_root=dataset_args['test_source_root'],
            target_roots=dataset_args['test_target_root'],
            transform=transforms_dict['transform_img_val'],
        )
        print(f"Number of test samples: {len(test_dataset)}")
        return test_dataset

    def find_favoured_channels(self):
        ce_weights, dice_weights = self.opts.ce_weights, self.opts.dice_weights
        if ce_weights is None and dice_weights is None:
            seg_weights = None
        elif ce_weights is not None and dice_weights is not None:
            if max(ce_weights) >= max(dice_weights):
                seg_weights = ce_weights
            else:
                seg_weights = dice_weights
        elif ce_weights is not None:
            seg_weights = ce_weights
        else:
            seg_weights = dice_weights
        
        if seg_weights is None:
            return np.array([True] * self.opts.label_nc)
        else:
            return np.isclose(seg_weights, max(seg_weights))
    
    def calc_loss(self, x, y, y_hat, where_msk_loss=None, weight_msk_loss=None):    
        if where_msk_loss is None:
            where_msk_loss = torch.ones(y.shape[0], dtype=torch.bool, device=y.device)
        assert torch.any(where_msk_loss) or self.opts.only_intra
        
        if self.opts.ce_weights is not None or self.opts.dice_weights is not None:
            favoured_channels = torch.tensor(self.find_favoured_channels(), dtype=torch.bool, device=y.device)
            weight_img_loss_where_msk_loss = y[where_msk_loss]
            weight_img_loss_where_msk_loss = weight_img_loss_where_msk_loss[:, favoured_channels]
            weight_img_loss_where_msk_loss = torch.any(weight_img_loss_where_msk_loss > 0.5, dim=1)
            weight_img_loss = torch.zeros(x.shape, dtype=bool, device=x.device)
            weight_img_loss[where_msk_loss] = weight_img_loss_where_msk_loss[:,None]
            weight_img_loss = torch.where(weight_img_loss, 1.0, 0.3)
        else:
            weight_img_loss = x * 0 + 1
        
        img_loss = dict(zip(["loss", "loss_dict", "id_logs"], self._img_loss(
            x*weight_img_loss, y_hat[:,:1]*weight_img_loss
        )))
        
        if weight_msk_loss is None:
            weight_msk_loss = torch.ones(x.shape, dtype=y.dtype, device=y.device)
            
        msk_loss = dict(zip(["loss", "loss_dict", "id_logs"], self._msk_loss(
            y[where_msk_loss]*weight_msk_loss[where_msk_loss], y_hat[where_msk_loss,1:]*weight_msk_loss[where_msk_loss]
        )))
        
        img_loss["loss_dict"]["loss"] = img_loss["loss_dict"]["loss_img"] + msk_loss["loss_dict"]["loss_msk"]
        return img_loss["loss"] + msk_loss["loss"], {**img_loss["loss_dict"], **msk_loss["loss_dict"]}, img_loss["id_logs"] 
    
    def _msk_loss(self, y, y_hat):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.ce_lambda > 0:
            loss_ce = self.ce_loss(y_hat, y)
            loss_dict['loss_ce'] = float(loss_ce)
            loss += loss_ce * self.opts.ce_lambda
        if self.opts.dice_lambda > 0:
            loss_dice = self.dice_loss(y_hat, y)
            loss_dict['loss_dice'] = float(loss_dice)
            loss += loss_dice * self.opts.dice_lambda
        loss_dict['loss_msk'] = float(loss)
        return loss, loss_dict, id_logs
    
    def _img_loss(self, y, y_hat):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.l2_lambda > 0:
            loss_l2 = self.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.ssim_lambda > 0:
            loss_ssim = self.ssim_loss(y_hat, y)
            loss_dict['loss_ssim'] = float(loss_ssim)
            loss += loss_l2 * self.opts.ssim_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        loss_dict['loss_img'] = float(loss)
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, label_nc=0, subscript=None, display_count=1, random_idx=None):
        im_data = []
        if random_idx is None:
            random_idx = [progress_count() % len(x)]*display_count
        for i in random_idx:
            cur_im_data = {
                'input_face': common.log_input_image(x[i], label_nc),
                'target_face': common.log_input_image(y[i], label_nc),
                'output_face': common.log_input_image(y_hat[i], label_nc),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript="{}{}".format(subscript + "_" if subscript is not None else "", random_idx[0]))
        return random_idx

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts),
            'global_step': self.global_step,
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.module.latent_avg#PARALLEL
        return save_dict
