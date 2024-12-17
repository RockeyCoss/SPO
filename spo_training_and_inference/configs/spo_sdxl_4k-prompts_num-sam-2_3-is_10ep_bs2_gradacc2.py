from configs.basic_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()

    ###### Model Setting ######
    config.pretrained.model = 'stabilityai/stable-diffusion-xl-base-1.0'
    config.pretrained.vae_model_name_or_path = 'madebyollin/sdxl-vae-fp16-fix'
    config.lora_rank = 64

    ###### Preference Model ######
    config.preference_model_func_cfg.ckpt_path = 'model_ckpts/sdxl_step-aware_preference_model.bin'

    ###### Compare Function ######
    config.compare_func_cfg.threshold = 0.4
    
    ###### Training ######
    config.sample.sample_batch_size = 2
    config.sample.num_inner_step = 3
    
    config.train.train_batch_size = 2
    config.train.learning_rate = 1e-5
    config.train.gradient_accumulation_steps = 2
    
    #### logging ####
    config.run_name = "spo_sdxl_4k-prompts_num-sam-2_3-is_10ep_bs2_gradacc2"
    
    return config
