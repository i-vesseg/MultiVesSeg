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
from configs.paths_config import model_paths
from datasets.images_dataset import ImagesDataset, MyRandomSampler
from models.psp import pSp
from training.ranger import Ranger

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
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda'
        self.opts.device = self.device

        # Initialize network
        self.net = pSp(self.opts)
        self.net = torch.nn.DataParallel(self.net, device_ids=[0, 1]).to(self.device)#PARALLEL

        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.net.module.latent_avg is None:
            with torch.no_grad():
                mean_latents = self.net.module.decoder.mean_latent(int(1e5), opts.src_tgt_domains, opts.n_domains)
                self.net.module.latent_avg = torch.nn.Parameter(mean_latents)#PARALLEL

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

        # Initialize optimizer
        self.optimizer, self.optimizer_seg = self.configure_optimizers()
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                        batch_size=self.opts.batch_size,
                                        sampler=MyRandomSampler(self.train_dataset, not self.opts.disable_balanced_sampling),
                                        num_workers=int(self.opts.workers),
                                        drop_last=True)
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
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def process_batch(self, batch):
        x, y, w, labels, has_gt = batch
        x, y, w = x.to(self.device).half(), y.to(self.device).half(), w.to(self.device).half()
        labels, has_gt = labels.to(self.device).half(), has_gt.to(self.device)
        
        is_swi = labels[:,1] > labels[:,0]
        is_tof = torch.logical_not(is_swi)
        is_semi = torch.logical_and(is_swi, has_gt)
        is_not_semi = torch.logical_and(is_swi, torch.logical_not(has_gt))
        #assert not torch.any(is_semi) #unsupervised
        #assert torch.all(x[is_swi, 1] == 1) (there is no brain neither vessel mask in swi)

        #TODO: Rethink of this as a general any class problem, it seems to be only 3 class
        #INTER DOMAIN BRANCH
        if not self.opts.only_intra and self.global_step % 2 == 1:
            #TODO: Probably the two branches are useless
            #UNSUPERVISED BRANCH
            if not torch.any(is_semi):
                y_hat = self.net(
                    x, torch.logical_not(labels).type(labels.dtype),
                    return_latents=False, feature_scale=min(1.0, 0.0001*self.global_step)
                )

                y_cycle = y_hat[:, :1].detach().clone()

                y_cycle, latent = self.net(
                    y_cycle, labels, return_latents=True, feature_scale=min(1.0, 0.0001*self.global_step), ExSeg=True
                )

                msk_loss_additional_tof = dict(zip(["loss", "loss_dict", "id_logs"], self._msk_loss(
                    y[is_tof], y_hat[is_tof,1:], None
                )))

                loss, loss_dict, id_logs = self.calc_loss(x, y, y_cycle, latent, is_swi, where_msk_loss=is_tof, weight_msk_loss=w)
                loss += msk_loss_additional_tof["loss"]

            #SEMI-SUPERVISED BRANCH
            else:
                y_hat = self.net(
                    x, torch.logical_not(labels).type(labels.dtype),
                    return_latents=False, feature_scale=min(1.0, 0.0001*self.global_step), is_semi=is_semi
                )

                y_cycle = y_hat[:, :1].detach().clone()

                y_cycle, latent = self.net(
                    y_cycle, labels, return_latents=True, feature_scale=min(1.0, 0.0001*self.global_step), ExSeg=True, is_semi=is_semi
                )

                msk_loss_additional_tof = dict(zip(["loss", "loss_dict", "id_logs"], self._msk_loss(
                    y[has_gt], y_hat[has_gt,1:], None
                )))

                loss, loss_dict, id_logs = self.calc_loss(x, y, y_cycle, latent, is_swi, where_msk_loss=has_gt, weight_msk_loss=w)
                loss += msk_loss_additional_tof["loss"]

        #INTRA DOMAIN BRANCH
        else:
            y_hat, latent = self.net(
                x, labels, return_latents=True, feature_scale=min(1.0, 0.0001*self.global_step), ExSeg=True
            )
            
            loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent, is_swi, where_msk_loss=has_gt, weight_msk_loss=w)

        assert id_logs is None
        return x.detach(), y.detach(), y_hat.detach(), loss, loss_dict, id_logs

    def train(self):
        self.net.train()
        finished_training = False
        while not finished_training:
            for batch in self.train_dataloader:
                self.optimizer_seg.zero_grad()
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    x, y, y_hat, loss, loss_dict, id_logs = self.process_batch(batch)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_seg)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Logging related
                if self.global_step % self.opts.image_interval == 0:
                    if True:#self.global_step == 0:
                        for random_idx in range(self.opts.batch_size):
                            self.parse_and_log_images(
                                id_logs, x, x, y_hat[:,:1], 'images/train/imgs', 0, random_idx=[random_idx]
                            )
                            self.parse_and_log_images(
                                id_logs, y, y, y_hat[:,1:], 'images/train/msks', self.opts.label_nc, random_idx=[random_idx]
                            ) 
                    else:
                        random_idx = self.parse_and_log_images(id_logs, x, x, y_hat[:,:1], 'images/train/imgs', 0)
                        self.parse_and_log_images(
                            id_logs, y, y, y_hat[:,1:], 'images/train/msks', self.opts.label_nc, random_idx=random_idx
                        )
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')
                
                #if self.global_step == 0:
                #    self.global_step += 1
                #    continue
                    
                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()

                    current_worst = self.best_val_loss[0][1]
                    current_dice = [v for k,v in val_loss_dict.items() if "dice_vessels" in k]
                    if any([dice > current_worst for dice in current_dice]):
                        current_dice = max(current_dice)
                        best_dice_ckpt = self.best_val_loss[0][0]
                        self.best_val_loss = sorted([(best_dice_ckpt, current_dice), *self.best_val_loss[1:]], key=lambda x: x[1])
                        self.checkpoint_me(val_loss_dict, is_best=True, best_path=best_dice_ckpt)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    finished_training = True
                    break

                self.global_step += 1

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        original = None
        predicted_inter = None
        predicted_intra = None
        curr_img_name = None
        translation = None
        metrics = {}
        for batch_idx, batch in enumerate(self.test_dataloader):
            x, y, w, labels, _ = batch
            if len(x) < 2:
                continue
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
                slice_prediction = y_hat[:,1:].clone() * slice_weight
                slice_original = y.clone() * slice_weight
                w = self.net.module.face_pool(w)
                y_hat = self.net.module.face_pool(y_hat)
                y = self.net.module.face_pool(y)
                
                loss_intra, cur_loss_dict_intra, id_logs_intra = self.calc_loss(x, y, y_hat, latent, is_swi, weight_msk_loss=w)

                # Logging related
                random_idx = self.parse_and_log_images(None, x, x, y_hat[:,:1], 
                    'images/test/imgs_intra', 0, subscript='{:04d}'.format(batch_idx),
                )
                self.parse_and_log_images(None, y, y, y_hat[:,1:], 
                    'images/test/msks_intra', self.opts.label_nc, subscript='{:04d}'.format(batch_idx), random_idx=random_idx
                )

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
                    
                    slice_prediction = y_hat[:,1:].clone() * slice_weight
                    slice_translation = y_hat[:,:1].clone() * slice_weight
                    y_hat = self.net.module.face_pool(y_hat)
                    
                    y_cycle = y_hat[:, :1].clone()
                    y_cycle, latent = self.net(
                        y_cycle, labels, return_latents=True, feature_scale=min(1.0, 0.0001*self.global_step)
                    )

                    # Logging related
                    self.parse_and_log_images(None, x, x, y_hat[:,:1],
                        'images/test/imgs_inter', 0, subscript='{:04d}'.format(batch_idx), random_idx=random_idx
                    )
                    self.parse_and_log_images(None, y, y, y_hat[:,1:], 
                        'images/test/msks_inter', self.opts.label_nc, subscript='{:04d}'.format(batch_idx), random_idx=random_idx
                    )

                    # Logging related
                    self.parse_and_log_images(None, x, x, y_cycle[:,:1],
                        'images/test/imgs_cycle', 0, subscript='{:04d}'.format(batch_idx), random_idx=random_idx
                    )
                    self.parse_and_log_images(None, y, y, y_cycle[:,1:], 
                        'images/test/msks_cycle', self.opts.label_nc, subscript='{:04d}'.format(batch_idx), random_idx=random_idx
                    )

                    loss_inter, cur_loss_dict_inter, id_logs_inter = self.calc_loss(
                        x, y, torch.cat([y_cycle[:,:1], y_hat[:,1:]], dim=1), latent, is_swi, weight_msk_loss=w
                    )

                    predicted_inter = np.concatenate([
                        predicted_inter, slice_prediction.cpu().numpy()
                    ]) if predicted_inter is not None else slice_prediction.cpu().numpy()
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

                volume_orig = np.argmax(volume_orig, axis=1)
                volume_intra = np.argmax(volume_intra, axis=1)
                if not self.opts.only_intra:
                    volume_inter = np.argmax(volume_inter, axis=1)
                    volume_ensemble = np.argmax(volume_ensemble, axis=1)

                if self.opts.label_nc == 3:
                    if not self.opts.consider_all_vessels:
                        roi = np.logical_not(np.isclose(volume_orig, 0))#experts have labeled only vessels inside the brain
                    else:
                        roi = np.full(volume_orig.shape, True, dtype=bool)#experts have labeled all vessels (within the weight mask)
                    vessels_orig = np.isclose(volume_orig, 2)
                    vessels_intra = np.logical_and(np.isclose(volume_intra, 2), roi)
                    if not self.opts.only_intra:
                        vessels_inter = np.logical_and(np.isclose(volume_inter, 2), roi)
                        vessels_ensemble = np.logical_and(np.isclose(volume_ensemble, 2), roi)
                else:
                    vessels_orig = volume_orig
                    vessels_intra = volume_intra
                    if not self.opts.only_intra:
                        vessels_inter = volume_inter
                        vessels_ensemble = volume_ensemble
                        
                #nib.save(nib.Nifti1Image(volume_orig.astype(float), None, None), f"{curr_img_name}_orig.nii.gz")
                #nib.save(nib.Nifti1Image(volume_inter.astype(float), None, None), f"{curr_img_name}_pred.nii.gz")
                #nib.save(nib.Nifti1Image(volume_intra.astype(float), None, None), f"{curr_img_name}_intra.nii.gz")
                #nib.save(nib.Nifti1Image(volume_ensemble.astype(float), None, None), f"{curr_img_name}_ensemble.nii.gz")
                #nib.save(nib.Nifti1Image(volume_trans.astype(float), None, None), f"{curr_img_name}_trans.nii.gz")

                metrics[curr_img_name] = {}
                metrics[curr_img_name]["intra_dice_vessels"] = DC(vessels_intra, vessels_orig)
                metrics[curr_img_name]["intra_hausdorff_vessels"] = HD(vessels_intra, vessels_orig)
                if not self.opts.only_intra:
                    metrics[curr_img_name]["inter_dice_vessels"] = DC(vessels_inter, vessels_orig)
                    metrics[curr_img_name]["inter_hausdorff_vessels"] = HD(vessels_inter, vessels_orig)
                    metrics[curr_img_name]["dice_vessels"] = DC(vessels_ensemble, vessels_orig)
                    metrics[curr_img_name]["hausdorff_vessels"] = HD(vessels_ensemble, vessels_orig)
                print(curr_img_name, metrics[curr_img_name])

                original = original[until:] if until is not None else None
                predicted_intra = predicted_intra[until:] if until is not None else None
                if not self.opts.only_intra:
                    predicted_inter = predicted_inter[until:] if until is not None else None
                    translation = translation[until:] if until is not None else None
                curr_img_name = img_names[-1] if until is not None else None

        metrics = {k: np.mean([metrics[patient_id][k] for patient_id in metrics]) for k in metrics[list(metrics.keys())[0]]}
        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        loss_dict = {**metrics, **loss_dict}
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
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

    def configure_optimizers(self):
        params = list(self.net.module.encoder.parameters())#PARALLEL
        if self.opts.train_decoder:
            params += list(self.net.module.decoder.parameters())#PARALLEL
        else:
            requires_grad(self.net.module.decoder, False)#PARALLEL
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        optimizer_seg = torch.optim.Adam(self.net.module.interpreter.classifiers.parameters(), lr=0.001)
        return optimizer, optimizer_seg

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        print(f'Loading dataset for {self.opts.dataset_type}')
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        #if dataset_args['train_target_root'] is not None and self.opts.only_intra:
        #    dataset_args['train_target_root'].pop('unlabeled', None)
        train_dataset = ImagesDataset(
            opts=self.opts,
            source_root=dataset_args['train_source_root'],
            target_roots=dataset_args['train_target_root'],
            source_transform=transforms_dict['transform_img_aug' if self.opts.use_da else 'transform_img'],
            target_transform=transforms_dict['transform_msk_aug' if self.opts.use_da else 'transform_msk'],
            one_target_slice=self.opts.one_target_slice,
        )
        test_dataset = ImagesDataset(
            opts=self.opts,
            source_root=dataset_args['val_source_root'],
            target_roots=dataset_args['val_target_root'],
            source_transform=transforms_dict['transform_img_val'],
            target_transform=transforms_dict['to_tensor'],
        )
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, latent, is_swi, where_msk_loss=None, weight_msk_loss=None):
        weight_img_loss = torch.where(
            is_swi,
            y_hat[:,-1:].permute(1, 2, 3, 0) * 0,
            (y[:,-1:].permute(1, 2, 3, 0) + 1) / 2,
        ).permute(3, 0, 1, 2)
        weight_img_loss = torch.where(weight_img_loss < 0.5, 0.3, 1.0)
        
        img_loss = dict(zip(["loss", "loss_dict", "id_logs"], self._img_loss(
            x*weight_img_loss, y_hat[:,:1]*weight_img_loss, latent
        )))
        
        if where_msk_loss is None:
            where_msk_loss = torch.ones(y.shape[0], dtype=torch.bool, device=y.device)
        assert torch.any(where_msk_loss) or self.opts.only_intra
        
        if weight_msk_loss is None:
            weight_msk_loss = torch.ones(x.shape, dtype=y.dtype, device=y.device)
            
        #opzione 1
        #y[weight_msk_loss.repeat(1, 3, 1, 1)==0] = y_hat[:,1:][weight_msk_loss.repeat(1, 3, 1, 1)==0]
        
        #opzione 2: detach
        
        #opzione 3: trova le size e seleziona con reshape
        
        #opzione 0: moltiplicazione
        msk_loss = dict(zip(["loss", "loss_dict", "id_logs"], self._msk_loss(
            y[where_msk_loss]*weight_msk_loss[where_msk_loss], y_hat[where_msk_loss,1:]*weight_msk_loss[where_msk_loss], latent
        )))
        
        img_loss["loss_dict"]["loss"] = img_loss["loss_dict"]["loss_img"] + msk_loss["loss_dict"]["loss_msk"]
        return img_loss["loss"] + msk_loss["loss"], {**img_loss["loss_dict"], **msk_loss["loss_dict"]}, img_loss["id_logs"] 
    
    def _msk_loss(self, y, y_hat, latent):
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
    
    def _img_loss(self, y, y_hat, latent):
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
