import ml_collections

def get_config():
    return basic_config()

def basic_config():
    config = ml_collections.ConfigDict()
    
    ###### General ######
    # random seed for reproducibility.
    config.seed = 42
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = None
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # whether or not to use xFormers to reduce memory usage.
    config.use_xformers = False
    # enable activation checkpointing or not. 
    # this reduces memory usage at the cost of some additional compute.
    config.use_checkpointing = False
    
    ###### Model Setting ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.use_lora = True
    config.lora_rank = 4
    
    ###### Preference Model ######
    config.preference_model_func_cfg = dict(
        type="step_aware_preference_model_func",
        model_pretrained_model_name_or_path='yuvalkirstain/PickScore_v1',
        processor_pretrained_model_name_or_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        ckpt_path='model_ckpts/sd-v1-5_step-aware_preference_model.bin',
    )
    
    ###### Compare Function ######
    config.compare_func_cfg = dict(
        type="preference_score_compare",
        threshold=0.3,
    )
    
    ##### dataset #####
    config.dataset_cfg = dict(
        type="PromptDataset",
        meta_json_path='assets/prompts/4k_training_prompts.json',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    )
    
    ##### dataloader ####
    config.dataloader_num_workers = 16
    config.dataloader_shuffle = True
    config.dataloader_pin_memory = True
    config.dataloader_drop_last = False

    ###### Training ######
    config.num_epochs = 10
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 20
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 5.0
    sample.sample_batch_size = 10
    # number of x_{t-1} sampled at each timestep.
    sample.num_sample_each_step = 2

    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training.
    train.train_batch_size = 10
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 6e-5
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = 1
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train.cfg = True

    train.divert_start_step = 4
    # coefficient of the KL divergence
    train.beta = 10.0
    # The coefficient constraining the probability ratio.
    train.eps = 0.1

    #### validation ####
    config.validation_prompts = ['A beautiful lake']
    config.num_validation_images = 2
    config.eval_interval = 1
    
    #### logging ####
    # run name for wandb logging and checkpoint saving.
    config.run_name = ""
    config.wandb_project_name = 'spo'
    config.wandb_entity_name = None
    # top-level logging directory for checkpoint saving.
    config.logdir = "work_dirs"
    config.save_interval = 1
    
    return config
