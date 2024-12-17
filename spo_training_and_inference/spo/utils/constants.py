import os

huggingface_cache_dir = os.environ.get('HUGGING_FACE_CACHE_DIR', None)
UNET_CKPT_NAME = "unet"
UNET_LORA_CKPT_NAME = "unet_lora.pt"
