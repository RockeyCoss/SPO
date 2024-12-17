from hydra.core.config_store import ConfigStore

from trainer.optimizers.adamw import AdamWOptimizerConfig
from trainer.optimizers.dummy_optimizer import (
    DummyOptimizerConfig,
    sd15_dummy_optimizer_cfg,
    sdxl_dummy_optimizer_cfg,
)

cs = ConfigStore.instance()
cs.store(group="optimizer", name="dummy", node=DummyOptimizerConfig)
cs.store(group="optimizer", name="sd15_dummy", node=sd15_dummy_optimizer_cfg)
cs.store(group="optimizer", name="sdxl_dummy", node=sdxl_dummy_optimizer_cfg)
cs.store(group="optimizer", name="adamw", node=AdamWOptimizerConfig)
