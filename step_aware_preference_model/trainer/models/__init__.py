from hydra.core.config_store import ConfigStore

from trainer.models.step_aware_preference_model import StepAwarePreferenceModelConfig

cs = ConfigStore.instance()
cs.store(group="model", name="spm", node=StepAwarePreferenceModelConfig)
