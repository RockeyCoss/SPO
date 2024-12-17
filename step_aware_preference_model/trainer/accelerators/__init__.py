from hydra.core.config_store import ConfigStore

from trainer.accelerators.debug_accelerator import DebugAcceleratorConfig
from trainer.accelerators.deepspeed_accelerator import (
    DeepSpeedAcceleratorConfig,
    sd15_deepspeed_accelerator_cfg,
    sdxl_deepspeed_accelerator_cfg,
)

ACCELERATOR_GROUP_NAME = "accelerator"

cs = ConfigStore.instance()
cs.store(group=ACCELERATOR_GROUP_NAME, name="deepspeed", node=DeepSpeedAcceleratorConfig)
cs.store(group=ACCELERATOR_GROUP_NAME, name="sd15_deepspeed", node=sd15_deepspeed_accelerator_cfg)
cs.store(group=ACCELERATOR_GROUP_NAME, name="sdxl_deepspeed", node=sdxl_deepspeed_accelerator_cfg)
cs.store(group=ACCELERATOR_GROUP_NAME, name="debug", node=DebugAcceleratorConfig)
