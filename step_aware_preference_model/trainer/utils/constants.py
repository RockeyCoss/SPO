import os

huggingface_cache_dir = os.environ.get('HUGGING_FACE_CACHE_DIR', None)
sd15_huggingface_path = 'runwayml/stable-diffusion-v1-5'
sdxl_huggingface_path = 'stabilityai/stable-diffusion-xl-base-1.0'
sdxl_vae_huggingface_path = 'madebyollin/sdxl-vae-fp16-fix'

sd15_model_type_name = 'sd1.5'
sdxl_model_type_name = 'sdxl'
