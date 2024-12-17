import collections
from dataclasses import dataclass

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from accelerate.logging import get_logger
from omegaconf import II
from transformers import (
    AutoProcessor, 
    AutoModel,
)
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.tasks.base_task import BaseTaskConfig, BaseTask
from trainer.utils.constants import (
    huggingface_cache_dir,
    sd15_huggingface_path,
    sdxl_huggingface_path,
    sdxl_vae_huggingface_path,
    sd15_model_type_name,
    sdxl_model_type_name,
)
from trainer.utils.batchable_ddim_scheduler import BatchableDDIMScheduler

logger = get_logger(__name__)


def numpy_to_pil(images):
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

@dataclass
class SPMTaskConfig(BaseTaskConfig):
    _target_: str = "trainer.tasks.spm_task.SPMTask"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    label_0_column_name: str = II("dataset.label_0_column_name")
    label_1_column_name: str = II("dataset.label_1_column_name")

    input_ids_column_name: str = II("dataset.input_ids_column_name")
    pixels_0_column_name: str = II("dataset.pixels_0_column_name")
    pixels_1_column_name: str = II("dataset.pixels_1_column_name")
    timesteps_column_name: str = "timesteps"
    
    model_type: str = II("dataset.model_type")
    use_pickscore_label: bool = False
    pickscore_threshold: float = 0.1
    cfg_scale: float = 5.0

    sdv15_input_ids_column_name: str = II("dataset.sdv15_input_ids_column_name")
    sdxl_input_ids_0_column_name: str = II("dataset.sdxl_input_ids_0_column_name")
    sdxl_input_ids_1_column_name: str = II("dataset.sdxl_input_ids_1_column_name")
    pil_img_0_column_name: str = II("dataset.pil_img_0_column_name")
    pil_img_1_column_name: str = II("dataset.pil_img_1_column_name")
    caption_column_name: str = II("dataset.caption_column_name")

    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    model_pretrained_model_name_or_path: str = II("model.model_pretrained_model_name_or_path")
    
    evaluation_timesteps: tuple[int] = (0, 251, 501, 701, 951)

sd15_spm_task_cfg = SPMTaskConfig()
sdxl_spm_task_cfg = SPMTaskConfig(use_pickscore_label=True)


class SPMTask(BaseTask):
    def __init__(self, cfg: SPMTaskConfig, accelerator: BaseAccelerator):
        super().__init__(cfg, accelerator)
        self.cfg = cfg
        
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
            cache_dir=huggingface_cache_dir,
        ).image_processor
        self.normalize = torchvision.transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std,
        )
        self.processed_img_size = (
            processor.size['shortest_edge'] if isinstance(processor.size, dict) 
            else processor.size
        )
        del processor
        
        if self.cfg.model_type == sd15_model_type_name:
            self.diff_pipe = StableDiffusionPipeline.from_pretrained(
                sd15_huggingface_path, 
                torch_dtype=torch.float16,
                cache_dir=huggingface_cache_dir,
            )
            self.diff_pipe.scheduler = BatchableDDIMScheduler.from_pretrained(
                sd15_huggingface_path, 
                subfolder="scheduler",
                cache_dir=huggingface_cache_dir,
            )
            self.diff_pipe.scheduler.set_timesteps(1000, device=self.accelerator.device)
            self.diff_pipe.scheduler.alphas_cumprod = self.diff_pipe.scheduler.alphas_cumprod.to(self.accelerator.device)
            self.diff_pipe.to(self.accelerator.device)
            self.diff_pipe.unet.eval()
            self.diff_pipe.unet.requires_grad_(False)
            self.diff_pipe.text_encoder.eval()
            self.diff_pipe.text_encoder.requires_grad_(False)
            self.diff_pipe.vae.eval()
            self.diff_pipe.vae.requires_grad_(False)

            self.sd_neg_prompt_embeds = self.diff_pipe.text_encoder(
                self.diff_pipe.tokenizer(
                    [""],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.diff_pipe.tokenizer.model_max_length,
                ).input_ids.to(self.accelerator.device)
            )[0]
        elif self.cfg.model_type == sdxl_model_type_name:
            self.diff_pipe = StableDiffusionXLPipeline.from_pretrained(
                sdxl_huggingface_path, 
                torch_dtype=torch.float16,
                cache_dir=huggingface_cache_dir,
            )
            self.diff_pipe.scheduler = BatchableDDIMScheduler.from_pretrained(
                sdxl_huggingface_path, 
                subfolder="scheduler",
                cache_dir=huggingface_cache_dir,
            )
            vae = AutoencoderKL.from_pretrained(
                sdxl_vae_huggingface_path,
                cache_dir=huggingface_cache_dir,
            )
            self.diff_pipe.vae = vae
            self.diff_pipe.scheduler.set_timesteps(1000, device=self.accelerator.device)
            self.diff_pipe.scheduler.alphas_cumprod = self.diff_pipe.scheduler.alphas_cumprod.to(self.accelerator.device)
            self.diff_pipe.to(self.accelerator.device, dtype=torch.float16)
            self.diff_pipe.unet.eval()
            self.diff_pipe.unet.requires_grad_(False)
            self.diff_pipe.text_encoder.eval()
            self.diff_pipe.text_encoder.requires_grad_(False)
            self.diff_pipe.text_encoder_2.eval()
            self.diff_pipe.text_encoder_2.requires_grad_(False)
            self.diff_pipe.vae.eval()
            self.diff_pipe.vae.requires_grad_(False)
            (
                _, 
                self.sd_neg_prompt_embeds, 
                _, 
                self.sd_neg_pooled_prompt_embeds,
            ) = self.diff_pipe.encode_prompt(
                prompt='',
                device=self.accelerator.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            self.add_time_ids = torch.tensor(
                [[1024, 1024, 0, 0, 1024, 1024]], 
                device=self.accelerator.device,
            )
        else:
            raise ValueError(f'Unsupported diffusion model type {self.cfg.model_type}')
        
        if self.cfg.use_pickscore_label:
            self.pickscore_processor = AutoProcessor.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                cache_dir=huggingface_cache_dir,
            )
            self.pickscore_model = AutoModel.from_pretrained(
                self.cfg.model_pretrained_model_name_or_path,
                cache_dir=huggingface_cache_dir,
            ).eval().to(self.accelerator.device)

    def train_step(self, model, criterion, batch):        
        with torch.no_grad():    
            timesteps = torch.randint(
                0, 
                self.diff_pipe.scheduler.config.num_train_timesteps, 
                (batch[self.cfg.pixels_0_column_name].size(0),), 
                device=batch[self.cfg.pixels_0_column_name].device,
            ).long()
            
            img_0, img_1 = self.get_noised_imgs(batch, timesteps)

            img_0 = (img_0 / 2 + 0.5).clamp(0, 1).float()
            img_1 = (img_1 / 2 + 0.5).clamp(0, 1).float()

            img_0 = F.interpolate(
                img_0, 
                size=(self.processed_img_size, self.processed_img_size), 
                mode='bicubic', 
                align_corners=False,
            )
            img_1 = F.interpolate(
                img_1, 
                size=(self.processed_img_size, self.processed_img_size), 
                mode='bicubic', 
                align_corners=False,
            )
            img_0 = img_0.clamp(0, 1)
            img_1 = img_1.clamp(0, 1)
            
            img_0 = self.normalize(img_0)
            img_1 = self.normalize(img_1)
            batch[self.cfg.pixels_0_column_name] = img_0
            batch[self.cfg.pixels_1_column_name] = img_1
            batch[self.cfg.timesteps_column_name] = timesteps
            
            if self.cfg.use_pickscore_label:
                self.update_as_pickscore_label(
                    batch,
                    self.cfg.pickscore_threshold,
                )

        loss = criterion(model, batch)
        return loss

    @staticmethod
    def features2probs(model, text_features, image_0_features, image_1_features):
        image_0_scores = model.logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_0_features))
        image_1_scores = model.logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_1_features))
        scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
        probs = torch.softmax(scores, dim=-1)
        image_0_probs, image_1_probs = probs[:, 0], probs[:, 1]
        return image_0_probs, image_1_probs

    @torch.no_grad()
    def valid_step(self, model, criterion, batch):
        image_0_features, image_1_features, text_features = criterion.get_features(
            model,
            batch[self.cfg.input_ids_column_name],
            batch[self.cfg.pixels_0_column_name],
            batch[self.cfg.pixels_1_column_name],
            batch[self.cfg.timesteps_column_name],
        )
        return self.features2probs(model, text_features, image_0_features, image_1_features)

    @staticmethod
    def pixel_values_to_pil_images(pixel_values):
        images = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = numpy_to_pil(images)
        return images

    def run_inference(self, model, criterion, dataloader, t):
        eval_dict = collections.defaultdict(list)
        logger.info("Running clip score...")
        for batch in dataloader:
            timesteps = torch.ones(
                batch[self.cfg.pixels_0_column_name].size(0),
                device=batch[self.cfg.pixels_0_column_name].device,
                dtype=torch.long,
            ) * t

            with torch.no_grad():
                if t > 0:
                    img_0, img_1 = self.get_noised_imgs(batch, timesteps)
                else:
                    img_0 = batch[self.cfg.pixels_0_column_name]
                    img_1 = batch[self.cfg.pixels_1_column_name]

                img_0 = (img_0 / 2 + 0.5).clamp(0, 1).float()
                img_1 = (img_1 / 2 + 0.5).clamp(0, 1).float()

                img_0 = F.interpolate(
                    img_0, 
                    size=(self.processed_img_size, self.processed_img_size), 
                    mode='bicubic', 
                    align_corners=False,
                )
                img_1 = F.interpolate(
                    img_1, 
                    size=(self.processed_img_size, self.processed_img_size), 
                    mode='bicubic', 
                    align_corners=False,
                )
                
                img_0 = img_0.clamp(0, 1)
                img_1 = img_1.clamp(0, 1)

                img_0 = self.normalize(img_0)
                img_1 = self.normalize(img_1)
                batch[self.cfg.pixels_0_column_name] = img_0
                batch[self.cfg.pixels_1_column_name] = img_1
                batch[self.cfg.timesteps_column_name] = timesteps
                if self.cfg.use_pickscore_label:
                    self.update_as_pickscore_label(
                        batch,
                        self.cfg.pickscore_threshold,
                    )
            
            image_0_probs, image_1_probs = self.valid_step(model, criterion, batch)
            if self.cfg.use_pickscore_label:
                win_lose_idx = batch[self.cfg.label_0_column_name] != 0.5
            else:
                win_lose_idx = slice(None)
            agree_on_0 = (image_0_probs > image_1_probs) * batch[self.cfg.label_0_column_name]
            agree_on_1 = (image_0_probs < image_1_probs) * batch[self.cfg.label_1_column_name]
            agree_on_0 = agree_on_0[win_lose_idx]
            agree_on_1 = agree_on_1[win_lose_idx]
            is_correct = agree_on_0 + agree_on_1
            eval_dict["is_correct"] += is_correct.tolist()
        return eval_dict

    @torch.no_grad()
    def evaluate(self, model, criterion, dataloader):
        all_metrics = {}
        for t in self.cfg.evaluation_timesteps:
            eval_dict = self.run_inference(model, criterion, dataloader, t)
            eval_dict = self.gather_dict(eval_dict)
            metrics = {
                f"accuracy on {t}": sum(eval_dict["is_correct"]) / len(eval_dict["is_correct"]),
                f"num_samples on {t}": len(eval_dict["is_correct"])
            }
            all_metrics.update(metrics)
        return all_metrics
    
    @torch.no_grad()
    def get_noised_imgs(self, batch, timesteps):
        img_0 = batch[self.cfg.pixels_0_column_name]
        img_1 = batch[self.cfg.pixels_1_column_name]
        
        if self.cfg.model_type == sd15_model_type_name:
            sd_prompt_embeds = self.diff_pipe.text_encoder(
                batch[self.cfg.sdv15_input_ids_column_name]
            )[0]
        elif self.cfg.model_type == sdxl_model_type_name:
            sd_prompt_embeds, sd_pooled_prompt_embeds = self.sdxl_encode_prompt_embeds(
                [self.diff_pipe.text_encoder, self.diff_pipe.text_encoder_2],
                text_input_ids_list=[
                    batch[self.cfg.sdxl_input_ids_0_column_name],
                    batch[self.cfg.sdxl_input_ids_1_column_name],
                ],
            )
            sd_neg_pooled_prompt_embeds = self.sd_neg_pooled_prompt_embeds.repeat(
                sd_prompt_embeds.shape[0], 1
            )
            add_time_ids = self.add_time_ids.to(
                dtype=sd_prompt_embeds.dtype
            ).repeat(sd_prompt_embeds.shape[0] * 2, 1)
            added_cond_kwargs = dict(
                text_embeds=torch.cat(
                    (
                        sd_neg_pooled_prompt_embeds, 
                        sd_pooled_prompt_embeds,
                    ),
                    dim=0
                ),
                time_ids=add_time_ids,
            )
        sd_neg_prompt_embeds = self.sd_neg_prompt_embeds.repeat(
            sd_prompt_embeds.shape[0], 1, 1
        )
        
        latents_0 = self.diff_pipe.vae.encode(
            img_0.to(torch.float16)
        ).latent_dist.sample()
        latents_0 = latents_0 * self.diff_pipe.vae.config.scaling_factor
        
        latents_1 = self.diff_pipe.vae.encode(
            img_1.to(torch.float16)
        ).latent_dist.sample()
        latents_1 = latents_1 * self.diff_pipe.vae.config.scaling_factor

        noise = torch.randn_like(latents_0)
        latents_0 = self.diff_pipe.scheduler.add_noise(latents_0, noise, timesteps)
        latents_1 = self.diff_pipe.scheduler.add_noise(latents_1, noise, timesteps)
        
        if self.cfg.model_type == sd15_model_type_name:
            noise_pred_0 = self.diff_pipe.unet(
                torch.cat([latents_0, latents_0]),
                torch.cat([timesteps, timesteps]),
                torch.cat([sd_neg_prompt_embeds, sd_prompt_embeds]),
            ).sample
            noise_pred_1 = self.diff_pipe.unet(
                torch.cat([latents_1, latents_1]),
                torch.cat([timesteps, timesteps]),
                torch.cat([sd_neg_prompt_embeds, sd_prompt_embeds]),
            ).sample
        elif self.cfg.model_type == sdxl_model_type_name:
            noise_pred_0 = self.diff_pipe.unet(
                torch.cat([latents_0, latents_0]),
                torch.cat([timesteps, timesteps]),
                torch.cat([sd_neg_prompt_embeds, sd_prompt_embeds]),
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            noise_pred_1 = self.diff_pipe.unet(
                torch.cat([latents_1, latents_1]),
                torch.cat([timesteps, timesteps]),
                torch.cat([sd_neg_prompt_embeds, sd_prompt_embeds]),
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        noise_pred_uncond_0, noise_pred_text_0 = noise_pred_0.chunk(2)
        noise_pred_0 = noise_pred_uncond_0 + self.cfg.cfg_scale * (
            noise_pred_text_0 - noise_pred_uncond_0
        )
        latents_0 = self.diff_pipe.scheduler.step(
            noise_pred_0, timesteps, latents_0, return_dict=True,
        ).pred_original_sample
        
        noise_pred_uncond_1, noise_pred_text_1 = noise_pred_1.chunk(2)
        noise_pred_1 = noise_pred_uncond_1 + self.cfg.cfg_scale * (
            noise_pred_text_1 - noise_pred_uncond_1
        )
        latents_1 = self.diff_pipe.scheduler.step(
            noise_pred_1, timesteps, latents_1, return_dict=True,
        ).pred_original_sample
        
        img_0 = self.diff_pipe.vae.decode(
                    latents_0 / self.diff_pipe.vae.config.scaling_factor, 
                    return_dict=False,
                )[0]
        img_1 = self.diff_pipe.vae.decode(
                    latents_1 / self.diff_pipe.vae.config.scaling_factor, 
                    return_dict=False,
                )[0]
        
        return img_0, img_1            
    
    @torch.no_grad()
    def update_as_pickscore_label(self, batch, threshold=0.1):
        device = batch[self.cfg.pixels_0_column_name].device
        pil_0_imgs = batch[self.cfg.pil_img_0_column_name]
        pil_1_imgs = batch[self.cfg.pil_img_1_column_name]
        captions = batch[self.cfg.caption_column_name]
                
        img_inputs_0 = self.pickscore_processor(
            images=pil_0_imgs,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        img_inputs_1 = self.pickscore_processor(
            images=pil_1_imgs,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        text_inputs = self.pickscore_processor(
            text=captions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        # embeddings
        # b, c
        image_embs_0 = self.pickscore_model.get_image_features(**img_inputs_0)
        image_embs_0 = image_embs_0 / torch.norm(image_embs_0, dim=-1, keepdim=True)

        image_embs_1 = self.pickscore_model.get_image_features(**img_inputs_1)
        image_embs_1 = image_embs_1 / torch.norm(image_embs_1, dim=-1, keepdim=True)
        
        # b, c
        text_embs = self.pickscore_model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        # batch version of scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        # b
        scores_0 = self.pickscore_model.logit_scale.exp() * (text_embs * image_embs_0).sum(-1)
        scores_1 = self.pickscore_model.logit_scale.exp() * (text_embs * image_embs_1).sum(-1)
        # b, 2
        scores = torch.stack((scores_0, scores_1), dim=1)
        probs = torch.softmax(scores, dim=-1)
        win_lose = probs[:, 0] - probs[:, 1] > threshold
        lose_win = probs[:, 1] - probs[:, 0] > threshold
        
        pickscore_label = scores_1.new_ones((scores_1.shape[0], 2)) * 0.5
        pickscore_label[win_lose] = pickscore_label.new_tensor([1, 0])
        pickscore_label[lose_win] = pickscore_label.new_tensor([0, 1])
        batch[self.cfg.label_0_column_name] = pickscore_label[:, 0]
        batch[self.cfg.label_1_column_name] = pickscore_label[:, 1]

    @staticmethod
    @torch.no_grad()
    def sdxl_encode_prompt_embeds(text_encoders, text_input_ids_list):
        prompt_embeds_list = []

        for i, text_encoder in enumerate(text_encoders):
            text_input_ids = text_input_ids_list[i]
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True, 
                return_dict=False,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds
