from dataclasses import dataclass

import torch
from torch import nn

from trainer.models.base_model import BaseModelConfig
from  trainer.models.time_conditioned_clip import HFTimeConditionedCLIPModel
from trainer.utils.constants import huggingface_cache_dir

@dataclass
class StepAwarePreferenceModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.step_aware_preference_model.StepAwarePreferenceModel"
    # for processor
    pretrained_model_name_or_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_model_name_or_path: str = "yuvalkirstain/PickScore_v1"


class StepAwarePreferenceModel(nn.Module):
    def __init__(self, cfg: StepAwarePreferenceModelConfig):
        super().__init__()
        self.model = HFTimeConditionedCLIPModel.from_pretrained(
            cfg.model_pretrained_model_name_or_path,
            cache_dir=huggingface_cache_dir,
        )

    def get_text_features(self, *args, **kwargs):
        return self.model.get_text_features(*args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)

    def forward(self, text_inputs=None, image_inputs=None, time_cond=None):
        outputs = ()
        if text_inputs is not None:
            outputs += self.model.get_text_features(text_inputs),
        if image_inputs is not None:
            outputs += self.model.get_image_features(image_inputs, time_cond),
        return outputs
    
    def init_adaln_paras(self):
        for layer in self.model.vision_model.encoder.layers:
            # init adaLN_modulation
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)
            bias = torch.zeros(6 * layer.embed_dim, dtype=layer.adaLN_modulation[-1].bias.dtype)
            with torch.no_grad():
                bias[2 * layer.embed_dim: 3 * layer.embed_dim] = 1
                bias[5 * layer.embed_dim:] = 1
                assert bias.shape == layer.adaLN_modulation[-1].bias.shape
                layer.adaLN_modulation[-1].bias = nn.Parameter(bias)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.model.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.model.t_embedder.mlp[2].weight, std=0.02)
        
        nn.init.constant_(self.model.vision_model.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.model.vision_model.adaLN_modulation[-1].bias, 0)

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def save(self, path):
        self.model.save_pretrained(path)
