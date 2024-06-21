from typing import Tuple, Union

import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep)
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0, self.alphas_cumprod.gather(0, prev_timestep), self.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance

def ddim_step_fetch_x0(
    self: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
) -> Union[DDIMSchedulerOutput, Tuple]:
    """
    Args:
        self: DDIMScheduler
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
    """
    assert isinstance(self, DDIMScheduler)
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )    
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep)
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0, self.alphas_cumprod.gather(0, prev_timestep), self.final_alpha_cumprod
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape)

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )
    return dict(
        pred_original_sample=pred_original_sample, 
        alpha_prod_t_prev=alpha_prod_t_prev, 
        pred_epsilon=pred_epsilon, 
        prev_timestep=prev_timestep,
    )

def ddim_step_fetch_x_t_1(
    self: DDIMScheduler,
    pred_original_sample,
    alpha_prod_t_prev,
    pred_epsilon,
    timestep,
    prev_timestep,
    dtype,
    num_sample_per_step,
    eta,
    generator=None,
):
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, pred_original_sample.shape).to(pred_original_sample.device)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if num_sample_per_step > 1:
        variance_noise = randn_tensor(
            (num_sample_per_step, *prev_sample_mean.shape), 
            generator=generator, 
            device=pred_original_sample.device, 
            dtype=dtype,
        )
        # num_sample_per_step, b, c, h, w
        prev_sample = prev_sample_mean[None] + std_dev_t[None] * variance_noise
    else:
        variance_noise = randn_tensor(
            prev_sample_mean.shape, generator=generator, device=pred_original_sample.device, dtype=dtype
        )
        # b,c,h,w
        prev_sample = prev_sample_mean + std_dev_t * variance_noise
    return prev_sample.type(dtype)
