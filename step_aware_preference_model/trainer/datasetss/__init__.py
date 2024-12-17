from hydra.core.config_store import ConfigStore

from trainer.datasetss.pick_a_pic_spm_dataset import (
    pick_a_pic_spm_sd15_dataset_cfg,
    pick_a_pic_spm_sdxl_dataset_cfg,
)

cs = ConfigStore.instance()
cs.store(group="dataset", name="pick_a_pic_spm_sd15", node=pick_a_pic_spm_sd15_dataset_cfg)
cs.store(group="dataset", name="pick_a_pic_spm_sdxl", node=pick_a_pic_spm_sdxl_dataset_cfg)
