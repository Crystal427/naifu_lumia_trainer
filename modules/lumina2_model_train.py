import safetensors
import torch
import os
import lightning as pl
import torch.nn.functional as F
from omegaconf import OmegaConf
from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger
from lightning.pytorch.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader
from IndexKits.index_kits.sampler import DistributedSamplerWithStartIndex, BlockDistributedSampler
from data_loader.arrow2_load_stream_ import TextImageArrowStream

from modules.lumina2_model import Lumina2Model
from models.lumina.transport import create_transport

import random
import math
def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = SupervisedFineTune(
        model_path=model_path, 
        config=config, 
        device=fabric.device
    )

    world_size = fabric.world_size
    dataset_name = config.dataset.get("name", "") if hasattr(config, "dataset") else ""
    dataloader = None

    # If dataset name points to a folder-based dataset (e.g., data.AspectRatioDataset / data.AdaptiveSizeDataset),
    # instantiate it dynamically; otherwise, fall back to Arrow loader.
    use_folder_dataset = isinstance(dataset_name, str) and dataset_name.startswith("data.")

    if use_folder_dataset:
        logger.info(f"loading folder dataset with class {dataset_name}")
        dataset_class = get_class(dataset_name)
        dataset = dataset_class(
            batch_size=config.trainer.batch_size,
            rank=fabric.global_rank,
            dtype=torch.float32,
            **config.dataset,
        )
        dataloader = dataset.init_dataloader()
    else:
        # Arrow dataset path (backward compatible)
        logger.info(f"loading arrow dataset from {config.dataset.index_file}")
        dataset = TextImageArrowStream(args="args",
                                       resolution=config.trainer.resolution,
                                       random_flip=config.dataset.random_flip,
                                       log_fn=logger.info,
                                       index_file=config.dataset.index_file,
                                       multireso=config.dataset.multireso,
                                       batch_size=config.trainer.batch_size,
                                       world_size=world_size
                                       )

        if config.dataset.multireso:
            sampler = BlockDistributedSampler(dataset, num_replicas=world_size, rank=fabric.global_rank, seed=config.trainer.seed,
                                              shuffle=True, drop_last=True, batch_size=config.trainer.batch_size)
        else:
            sampler = DistributedSamplerWithStartIndex(dataset, num_replicas=world_size, rank=fabric.global_rank, seed=config.trainer.seed,
                                                       shuffle=True, drop_last=True)
        
        dataloader = DataLoader(dataset, batch_size=config.trainer.batch_size, shuffle=False, sampler=sampler,
                            num_workers=config.dataset.num_workers, pin_memory=True, drop_last=True)
    


    params_to_optim = [{"params": model.parameters()}]
    if config.advanced.get("train_text_encoder"):
        lr = config.advanced.get("text_encoder_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.text_encoder.parameters(), "lr": lr})

    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(params_to_optim, **optim_param)
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(
            optimizer, **config.scheduler.params
        )
        
    if config.trainer.get("resume"):
        latest_ckpt = get_latest_checkpoint(config.trainer.checkpoint_dir)
        remainder = {}
        if latest_ckpt:
            logger.info(f"Loading weights from {latest_ckpt}")
            remainder = sd = load_torch_file(ckpt=latest_ckpt, extract=False)
            if latest_ckpt.endswith(".safetensors"):
                remainder = safetensors.safe_open(latest_ckpt, "pt").metadata()
            model.load_state_dict(sd.get("state_dict", sd))
            config.global_step = remainder.get("global_step", 0)
            config.current_epoch = remainder.get("current_epoch", 0)


    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    model.model, optimizer = fabric.setup(model.model, optimizer)
    model.model.mark_forward_method("forward_with_cfg")

    if hasattr(model, "setup"):
        model.setup(fabric)
    
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


class SupervisedFineTune(Lumina2Model):
    def get_module(self):
        return self.model
    
    
    def get_similar_size(self, base_size):
        #获得差最小的尺寸
        base_size_list = [1024*1024, 512*512, 768*768, 1280*1280, 1536*1536]
        min_diff = float('inf')
        target_size = base_size
        for size in base_size_list:
            diff = abs(size - base_size)
            if diff < min_diff:
                min_diff = diff
                target_size = size
        return target_size
    
    def forward(self, batch):
        # base_size = batch["base_size"][0]
        images = batch["pixels"].to(self.target_device)       
        prompts = batch["prompts"]
        # target_size = base_size
        # if base_size > 1024*1024:
        #     if random.random() < 0.5:
        #         target_size = 1024*1024
        
        #         mode_list = ['nearest', 'bilinear', 'bicubic', 'area']
        #         mode = random.choice(mode_list)
        #         scale_factor = math.sqrt(target_size/base_size)
        #         images = F.interpolate(images, scale_factor=scale_factor,
        #                             mode=mode, align_corners=None if mode in ['nearest', 'area'] else False)

        # for train_res in self.config.advanced.get("train_res", [1024]):
            
            
            # if base_size > 1024*1024:
            #     if random.random() < 0.5:
            #         target_size = 1024*1024
            
            #         mode_list = ['nearest', 'bilinear', 'bicubic', 'area']
            #         mode = random.choice(mode_list)
            #         scale_factor = 0.75
            #         images = F.interpolate(images, scale_factor=scale_factor,
            #                             mode=mode, align_corners=None if mode in ['nearest', 'area'] else False)

        # NOTE: seq_len 影响 time shift（mu）的计算，应与 token 数一致。
        # 在 Lumina 的实现中，mu 使用 (h//2)*(w//2)（latent 维度的一半作为 patch 大小 2 的 token 网格）。
        # 因此需要在获得 latent 尺寸后动态设置，而不是固定为 4096（仅适用于 1024x1024）。

        
        # 编码文本提示
        prompt_embeds, prompt_masks = self.encode_prompt(
            prompts, 
            self.text_encoder,
            self.tokenizer,
            proportion_empty_prompts=0.1
        )

        # 对图像进行VAE编码
        latents = self.encode_images(images)  # [B, C, H, W]（list of tensors）

        # 动态创建 transport，确保 seq_len 与实际 token 数一致
        # token_count = (latent_h / patch_size) * (latent_w / patch_size)
        patch_size = getattr(self, "model_patch_size", 2)
        sample_latent = latents[0] if isinstance(latents, (list, tuple)) else latents
        latent_h, latent_w = sample_latent.shape[-2], sample_latent.shape[-1]
        token_h = latent_h // patch_size
        token_w = latent_w // patch_size
        seq_len = int(token_h * token_w)

        trans = create_transport(
            "Linear",
            "velocity",
            None,
            None,
            None,
            snr_type=self.config.advanced.snr_type,
            do_shift=not self.config.advanced.no_shift,
            seq_len=seq_len,
        )

        # muti resolution
        # if len(latents.shape) == 3:
        #     latents = latents.unsqueeze(0)
        # muti resolution
        # latents_mb_256 = [self.apply_average_pool(x, 4) for x in latents]

        model_kwargs = dict(cap_feats=prompt_embeds, cap_mask=prompt_masks)
        loss_dict = trans.training_losses(self.model, latents, model_kwargs)
        # loss_dict_256 = trans.training_losses(self.model, latents_mb_256, model_kwargs)

        loss_1024 = loss_dict["loss"].sum() / self.batch_size
        # loss_256 = loss_dict_256["loss"].sum() / self.batch_size
        loss = loss_1024 

        # 记录训练损失
        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_1024", loss_1024, prog_bar=True)
        # self.log("loss_256", loss_256, prog_bar=True)
        
        # 添加梯度裁剪
        if hasattr(self.config.trainer, 'grad_clip') and self.config.trainer.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm=self.config.trainer.grad_clip
            )
            self.log("grad_norm", grad_norm, prog_bar=True)
            
        return loss
