from mmengine import Registry, build_from_cfg

DATASETS = Registry('dataset')

def build_dataset(cfg):
    return build_from_cfg(cfg, DATASETS)
