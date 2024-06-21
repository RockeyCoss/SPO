from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
from .ddim_seperate import ddim_step_fetch_x0, ddim_step_fetch_x_t_1


@torch.no_grad()
def multi_sample_pipeline(
    self: StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    
    divert_start_step=0,
    num_samples_each_step=2,
    preference_model_fn=None,
    compare_fn=None,
    extra_info=None,
    **kwargs,
):
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )
    log_prompt_embeds = prompt_embeds
    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

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

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    
    all_prev_latents = None
    current_latents = None
    if divert_start_step == 0:
        current_latents = latents
    last_timestep = None

    denoise_idx = None
    
    valid_timesteps = []
    valid_current_latents = []
    valid_next_latents = []
    valid_prompt_embeds = []
    preference_score_logs = []
    
    with self.progress_bar(total=timesteps.shape[0]) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
            
            if i >= divert_start_step:
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
                    if do_classifier_free_guidance:
                        prompt_embeds = torch.cat([
                            negative_prompt_embeds.repeat(num_samples_each_step, 1, 1),
                            log_prompt_embeds.repeat(num_samples_each_step, 1, 1),
                        ])
                elif i > divert_start_step:
                    # num_sample_per_step*b, c, h, w
                    pred_x0 = pred_dict['pred_original_sample']
                    pred_x0 = self.vae.decode(
                        pred_x0.to(self.vae.dtype) / self.vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]
                    preference_timestep = t.repeat(pred_x0.shape[0])
                    extra_info['timesteps'] = preference_timestep
                    # num_sample_per_step*b
                    # preference scores of i-1
                    preference_scores = preference_model_fn(pred_x0, extra_info)
                    preference_score_logs.append(preference_scores)
                    preference_scores = preference_scores.reshape(num_samples_each_step, -1)
                    # indices: 2,b
                    # valid_samples: b
                    indices, valid_samples = compare_fn(preference_scores)
                    
                    # 2,valid_num,c,h,w
                    valid_next_latents.append(torch.gather(
                        all_prev_latents, 
                        dim=0, 
                        index=indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *all_prev_latents.shape[2:]),
                    )[valid_samples.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                        2, -1, *all_prev_latents.shape[2:]
                    )].reshape(2, -1, *all_prev_latents.shape[2:]))
                    # valid_num,1,c,h,w
                    valid_current_latents.append(current_latents[valid_samples].unsqueeze(1))
                    # valid_num,1
                    valid_timesteps.append(last_timestep.repeat(valid_current_latents[-1].shape[0]).unsqueeze(1))
                    # valid_num,1,l,c
                    valid_prompt_embeds.append(log_prompt_embeds[valid_samples].unsqueeze(1))
                    
                    # b->1,b,c,h,w
                    denoise_idx = torch.randint(
                        0, num_samples_each_step, 
                        size=(all_prev_latents.shape[1],),
                        device=all_prev_latents.device,
                    )[None, :, None, None, None].expand(-1, -1, *all_prev_latents.shape[2:])
                        
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
                    # b,c,h,w
                    current_latents = torch.gather(
                        all_prev_latents,
                        dim=0,
                        index=denoise_idx,
                    )[0]

                    prev_latents = ddim_step_fetch_x_t_1(
                        self.scheduler,
                        dtype=latents.dtype,
                        num_sample_per_step=num_samples_each_step,
                        timestep=t,
                        **extra_step_kwargs,
                        **pred_dict,
                    )

                latents = prev_latents.flatten(0, 1)
                all_prev_latents = prev_latents
                last_timestep = t
                # x0
                if i == len(timesteps) - 1:
                    # num_sample_per_step*b, c, h, w 
                    prev_latents = self.vae.decode(
                        prev_latents.flatten(0, 1).to(self.vae.dtype) / self.vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]
                    preference_timestep = torch.zeros_like(t).repeat(prev_latents.shape[0])
                    extra_info['timesteps'] = preference_timestep
                    # num_sample_per_step*b
                    # preference scores of 0
                    preference_scores = preference_model_fn(prev_latents, extra_info)
                    preference_score_logs.append(preference_scores)
                    preference_scores = preference_scores.reshape(num_samples_each_step, -1)
                    # indices: 2,b
                    # valid_samples: b
                    indices, valid_samples = compare_fn(preference_scores)
                    # 2,valid_num,c,h,w
                    valid_next_latents.append(torch.gather(
                        all_prev_latents, 
                        dim=0, 
                        index=indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *all_prev_latents.shape[2:]),
                    )[valid_samples.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                        2, -1, *all_prev_latents.shape[2:]
                    )].reshape(2, -1, *all_prev_latents.shape[2:]))
                    # valid_num,1,c,h,w
                    valid_current_latents.append(current_latents[valid_samples].unsqueeze(1))
                    # valid_num,1
                    valid_timesteps.append(last_timestep.repeat(valid_current_latents[-1].shape[0]).unsqueeze(1))
                    # valid_num,1,l,c
                    valid_prompt_embeds.append(log_prompt_embeds[valid_samples].unsqueeze(1))                    
            else:
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
                if i == divert_start_step - 1:
                    current_latents = latents

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
    # valid_num, 1
    valid_timesteps = torch.cat(valid_timesteps, dim=0)
    # valid_num, 1, c, h, w
    valid_current_latents = torch.cat(valid_current_latents, dim=0)
    # valid_num,2,c,h,w
    valid_next_latents = torch.cat(valid_next_latents, dim=1).transpose(0, 1).contiguous()
    # valid_num,1,l,c
    valid_prompt_embeds = torch.cat(valid_prompt_embeds, dim=0)
    
    preference_score_logs = torch.cat(preference_score_logs, dim=0)
    
    return valid_timesteps, valid_current_latents, valid_next_latents, valid_prompt_embeds, preference_score_logs
