import argparse
import os.path as osp
import sys
# Add the project directory to the Python path to simplify imports without manually setting PYTHONPATH.
sys.path.insert(
    0, osp.abspath(
        osp.join(osp.dirname(osp.abspath(__file__)), "..")
    ),
)
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from spo.utils import huggingface_cache_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_id', default='SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep')
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--prompt', 
        default='a child and a penguin sitting in front of the moon',
    )
    parser.add_argument(
        '--cfg_scale',
        default=5.0,
        type=float,
    )
    parser.add_argument(
        '--output_filename',
        default='spo_sdxl_img.png',
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
    )
    args = parser.parse_args()
    
    ckpt_id = args.ckpt_id
    inference_dtype = torch.float16
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        ckpt_id, 
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir,
    )
    vae = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix',
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir,
    )
    pipe.vae = vae
    pipe.to(args.device)
    
    generator=torch.Generator(device=args.device).manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        guidance_scale=args.cfg_scale,
        generator=generator,
        output_type='pil',
    ).images[0]
    image.save(args.output_filename)

if __name__ == '__main__':
    main()