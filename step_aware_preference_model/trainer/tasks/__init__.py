
from hydra.core.config_store import ConfigStore

from trainer.tasks.spm_task import (
    sd15_spm_task_cfg,
    sdxl_spm_task_cfg,
)

cs = ConfigStore.instance()
cs.store(group="task", name="sd15_spm", node=sd15_spm_task_cfg)
cs.store(group="task", name="sdxl_spm", node=sdxl_spm_task_cfg)
