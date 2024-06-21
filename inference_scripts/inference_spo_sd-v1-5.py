import argparse
import torch
from diffusers import StableDiffusionPipeline
from spo.utils import huggingface_cache_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_id', default='SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep')
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--prompt', 
        default='an image of a beautiful lake',
    )
    parser.add_argument(
        '--cfg_scale',
        default=7.5,
        type=float,
    )
    parser.add_argument(
        '--output_filename',
        default='spo_sdv1-5_img.png',
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
    )
    args = parser.parse_args()
    
    ckpt_id = args.ckpt_id
    inference_dtype = torch.float16
    
    pipe = StableDiffusionPipeline.from_pretrained(
        ckpt_id, 
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir,
    )
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