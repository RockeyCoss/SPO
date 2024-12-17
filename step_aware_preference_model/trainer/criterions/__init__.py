from hydra.core.config_store import ConfigStore

from trainer.criterions.spm_criterion import SPMCriterionConfig

cs = ConfigStore.instance()
cs.store(group="criterion", name="spm", node=SPMCriterionConfig)
