from configs.basic_config import basic_config

def get_config():
    return exp_config()

def exp_config():
    config = basic_config()
    
    ###### Training ######
    config.sample.num_sample_each_step = 4
    
    #### logging ####
    config.run_name = "spo_sd-v1-5_4k-prompts_num-sam-4_10ep_bs10"

    return config
