from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    rescale_noise_cfg,
    retrieve_timesteps,
    is_torch_xla_available,
)
from .ddim_seperate import ddim_step_fetch_x0, ddim_step_fetch_x_t_1
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


@torch.no_grad()
def multi_sample_pipeline_sdxl(
    self: StableDiffusionXLPipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    denoising_end: Optional[float] = None,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    callback=None,
    callback_steps=None,
    
    divert_start_step=0,
    num_samples_each_step=2,
    num_inner_step=0,
    preference_model_fn=None,
    compare_fn=None,
    extra_info=None,
    **kwargs,
):
    # 0. Default height and width to unet
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._denoising_end = denoising_end
    self._interrupt = False


    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )
    log_prompt_embeds = prompt_embeds
    log_add_text_embeds = pooled_prompt_embeds

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )


    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

    add_time_ids = self._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = self._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids
    log_add_time_ids = add_time_ids.to(device)

    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
    negative_add_time_ids = negative_add_time_ids.to(device)

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    
    # ignored 8.1 Apply denoising_end
    
    # 9. Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    self._num_timesteps = len(timesteps)
    
    denoise_idx = None
    
    valid_timesteps = []
    valid_current_latents = []
    valid_next_latents = []
    valid_prompt_embeds = []
    valid_add_text_embeds = []
    preference_score_logs = []
    inner_step_left = 0
    
    timestep_cache = []
    current_latents_cache = []
    next_latents_cache = []
    
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
            
            if (i >= divert_start_step and inner_step_left == 0) or i == len(timesteps) - 1:
                inner_step_left = num_inner_step
                # pred_x0: (num_sample_per_step/1)*b, c, h, w 
                pred_dict = ddim_step_fetch_x0(
                    self.scheduler,
                    noise_pred,
                    t,
                    latents,
                )
                if i == divert_start_step:
                    # prev_latents: num_sample_per_step, b, c, h, w
                    prev_latents = ddim_step_fetch_x_t_1(
                        self.scheduler,
                        dtype=latents.dtype,
                        num_sample_per_step=num_samples_each_step,
                        timestep=t,
                        **extra_step_kwargs,
                        **pred_dict,
                    )
                    if self.do_classifier_free_guidance:
                        prompt_embeds = torch.cat([
                            negative_prompt_embeds.repeat(num_samples_each_step, 1, 1),
                            log_prompt_embeds.repeat(num_samples_each_step, 1, 1),
                        ])
                        add_text_embeds = torch.cat([
                            negative_pooled_prompt_embeds.repeat(num_samples_each_step, 1),
                            log_add_text_embeds.repeat(num_samples_each_step, 1),
                        ], dim=0)
                        add_time_ids = torch.cat([
                            negative_add_time_ids,
                            log_add_time_ids,
                        ], dim=0).repeat(num_samples_each_step * batch_size * num_images_per_prompt, 1)
                    
                    # b,c,h,w->num_sample_per_step, b, c, h, w
                    current_latents_cache.append(latents.unsqueeze(0).repeat(num_samples_each_step, 1, 1, 1, 1))
                    next_latents_cache.append(prev_latents)
                    timestep_cache.append(t)
                    
                    latents = prev_latents.flatten(0, 1)
                elif i > divert_start_step and i != len(timesteps) - 1:
                    # num_sample_per_step*b, c, h, w
                    pred_x0 = pred_dict['pred_original_sample']
                    pred_x0 = self.vae.decode(
                        pred_x0.to(self.vae.dtype) / self.vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]
                    reward_timestep = t.repeat(pred_x0.shape[0])
                    extra_info['timesteps'] = reward_timestep
                    # num_sample_per_step*b
                    # preference scores of i-1
                    preference_scores = preference_model_fn(pred_x0, extra_info)
                    preference_score_logs.append(preference_scores)
                    preference_scores = preference_scores.reshape(num_samples_each_step, -1)
                    # indices: 2,b
                    # valid_samples: b
                    indices, valid_samples = compare_fn(preference_scores)
                    
                    # num_sample,b,num_inner_step,c,h,w
                    next_latents_cache = torch.stack(next_latents_cache, dim=2)
                    
                    indices = indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                        -1, -1, *next_latents_cache.shape[2:]
                    )
                    
                    # after gather: 2,b,num_inner_step,c,h,w
                    # after index: 2,valid_num,num_inner_step,c,h,w
                    collected_valid_next_latents = torch.gather(
                        next_latents_cache, 
                        dim=0, 
                        index=indices,
                    )[valid_samples.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                        2, -1, *next_latents_cache.shape[2:]
                    )].reshape(2, -1, *next_latents_cache.shape[2:])
                    # 2,valid_num,c,h,w   num_inner_step entires
                    valid_next_latents.extend(
                        [
                            collected_valid_next_latents[:,:,i] for i in range(collected_valid_next_latents.size(2))
                        ]
                    )
                    
                    # num_sample,b,num_inner_step,c,h,w
                    current_latents_cache = torch.stack(current_latents_cache, dim=2)
                    # 2,valid_num,num_inner_step,c,h,w
                    collected_valid_current_latents = torch.gather(
                        current_latents_cache,
                        dim=0,
                        index=indices,
                    )[valid_samples.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                        2, -1, *current_latents_cache.shape[2:]
                    )].reshape(2, -1, *current_latents_cache.shape[2:])
                    # 2,valid_num,c,h,w   num_inner_step entires
                    valid_current_latents.extend(
                        [
                            collected_valid_current_latents[:,:,i] for i in range(collected_valid_current_latents.size(2))
                        ]
                    )
                    # valid_num,1   num_inner_step entires
                    valid_timesteps.extend([item.repeat(valid_current_latents[-1].shape[1]).unsqueeze(1) for item in timestep_cache])
                    # valid_num,1,l,c   num_inner_step entires
                    valid_prompt_embeds.extend([log_prompt_embeds[valid_samples].unsqueeze(1)] * collected_valid_next_latents.size(2))
                    # valid_num,1,c   num_inner_step entires
                    valid_add_text_embeds.extend([log_add_text_embeds[valid_samples].unsqueeze(1)] * collected_valid_next_latents.size(2))
                    
                    # b->1,b,c,h,w
                    denoise_idx = torch.randint(
                        0, num_samples_each_step, 
                        size=(current_latents_cache.shape[1],),
                        device=current_latents_cache.device,
                    )[None, :, None, None, None].expand(-1, -1, *current_latents_cache.shape[3:])
                    
                    for k, v in pred_dict.items():
                        if k != 'prev_timestep':
                            v = v.reshape(
                                num_samples_each_step, -1, *v.shape[1:]
                            )
                            # b,c,h,w
                            v = torch.gather(
                                v,
                                dim=0,
                                index=denoise_idx,
                            )[0]
                            pred_dict[k] = v

                    timestep_cache = []
                    current_latents_cache = []
                    next_latents_cache = []

                    # b,c,h,w
                    current_latents = torch.gather(
                        latents.reshape(num_samples_each_step, -1, *latents.shape[1:]),
                        dim=0,
                        index=denoise_idx,
                    ).repeat(num_samples_each_step, 1, 1, 1, 1)
                    current_latents_cache.append(current_latents)

                    prev_latents = ddim_step_fetch_x_t_1(
                        self.scheduler,
                        dtype=latents.dtype,
                        num_sample_per_step=num_samples_each_step,
                        timestep=t,
                        **extra_step_kwargs,
                        **pred_dict,
                    )
                    next_latents_cache.append(prev_latents)
                    timestep_cache.append(t)
                    
                    latents = prev_latents.flatten(0, 1)
                else:
                    # handle last step
                    prev_latents = ddim_step_fetch_x_t_1(
                        self.scheduler,
                        dtype=latents.dtype,
                        num_sample_per_step=1,
                        timestep=t,
                        **extra_step_kwargs,
                        **pred_dict,
                    )
                    current_latents_cache.append(latents.reshape(num_samples_each_step, -1, *latents.shape[1:]))
                    next_latents_cache.append(prev_latents.reshape(num_samples_each_step, -1, *prev_latents.shape[1:]))
                    timestep_cache.append(t)

                    # num_sample_per_step*b, c, h, w 
                    prev_latents = self.vae.decode(
                        prev_latents.to(self.vae.dtype) / self.vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]
                    reward_timestep = torch.zeros_like(t).repeat(prev_latents.shape[0])
                    extra_info['timesteps'] = reward_timestep
                    # num_sample_per_step*b
                    # preference scores of i-1
                    preference_scores = preference_model_fn(prev_latents, extra_info)
                    preference_score_logs.append(preference_scores)
                    preference_scores = preference_scores.reshape(num_samples_each_step, -1)
                    # indices: 2,b
                    # valid_samples: b
                    indices, valid_samples = compare_fn(preference_scores)
                    
                    # num_sample,b,num_inner_step,c,h,w
                    next_latents_cache = torch.stack(next_latents_cache, dim=2)
                    
                    indices = indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                        -1, -1, *next_latents_cache.shape[2:]
                    )
                    
                    # after gather: 2,b,num_inner_step,c,h,w
                    # after index: 2,valid_num,num_inner_step,c,h,w
                    collected_valid_next_latents = torch.gather(
                        next_latents_cache, 
                        dim=0, 
                        index=indices,
                    )[valid_samples.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                        2, -1, *next_latents_cache.shape[2:]
                    )].reshape(2, -1, *next_latents_cache.shape[2:])
                    # 2,valid_num,c,h,w   num_inner_step entires
                    valid_next_latents.extend(
                        [
                            collected_valid_next_latents[:,:,i] for i in range(collected_valid_next_latents.size(2))
                        ]
                    )
                    
                    # num_sample,b,num_inner_step,c,h,w
                    current_latents_cache = torch.stack(current_latents_cache, dim=2)
                    # 2,valid_num,num_inner_step,c,h,w
                    collected_valid_current_latents = torch.gather(
                        current_latents_cache,
                        dim=0,
                        index=indices,
                    )[valid_samples.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                        2, -1, *current_latents_cache.shape[2:]
                    )].reshape(2, -1, *current_latents_cache.shape[2:])
                    # 2,valid_num,c,h,w   num_inner_step entires
                    valid_current_latents.extend(
                        [
                            collected_valid_current_latents[:,:,i] for i in range(collected_valid_current_latents.size(2))
                        ]
                    )
                    # valid_num,1   num_inner_step entires
                    valid_timesteps.extend([item.repeat(valid_current_latents[-1].shape[1]).unsqueeze(1) for item in timestep_cache])
                    # valid_num,1,l,c   num_inner_step entires
                    valid_prompt_embeds.extend([log_prompt_embeds[valid_samples].unsqueeze(1)] * collected_valid_next_latents.size(2))
                    # valid_num,1,c   num_inner_step entires
                    valid_add_text_embeds.extend([log_add_text_embeds[valid_samples].unsqueeze(1)] * collected_valid_next_latents.size(2))
            else:
                latents_cp = latents
                pred_dict = ddim_step_fetch_x0(
                    self.scheduler,
                    noise_pred, 
                    t, 
                    latents, 
                )
                latents = ddim_step_fetch_x_t_1(
                    self.scheduler,
                    dtype=latents.dtype,
                    num_sample_per_step=1,
                    timestep=t,
                    **extra_step_kwargs,
                    **pred_dict,
                )
                if i >= divert_start_step:
                    inner_step_left -= 1
                    current_latents_cache.append(latents_cp.reshape(num_samples_each_step, -1, *latents_cp.shape[1:]))
                    next_latents_cache.append(latents.reshape(num_samples_each_step, -1, *latents.shape[1:]))
                    timestep_cache.append(t)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

            if XLA_AVAILABLE:
                xm.mark_step()
    # valid_num, 1
    valid_timesteps = torch.cat(valid_timesteps, dim=0)
    # valid_num,2,c,h,w
    valid_current_latents = torch.cat(valid_current_latents, dim=1).transpose(0, 1).contiguous()
    # valid_num,2,c,h,w
    valid_next_latents = torch.cat(valid_next_latents, dim=1).transpose(0, 1).contiguous()
    # valid_num,1,l,c
    valid_prompt_embeds = torch.cat(valid_prompt_embeds, dim=0)
    # valid_num,1,c
    valid_add_text_embeds = torch.cat(valid_add_text_embeds, dim=0)
    preference_score_logs = torch.cat(preference_score_logs, dim=0)
    return (
        valid_timesteps, 
        valid_current_latents, 
        valid_next_latents, 
        valid_prompt_embeds,
        valid_add_text_embeds, 
        negative_add_time_ids, 
        log_add_time_ids,
        preference_score_logs
    )
