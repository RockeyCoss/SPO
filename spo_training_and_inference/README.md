# SPO Training and Inference Code

This folder contains the code for SPO training and inference.

## Installation
1. Pull the Docker Image
```bash
sudo docker pull rockeycoss/spo:v1
```
2. Run the Docker Container and Enter It
```bash
sudo docker run --gpus all -it --ipc=host rockeycoss/spo:v1 /bin/bash
```
3. Clone the Repository
```bash
git clone https://github.com/RockeyCoss/SPO
cd ./SPO/spo_training_and_inference
```
4. Login to wandb
```bash
wandb login {Your wandb key}
```
5. (Optional) To customize the location for saving models downloaded from Hugging Face, you can use the following command:
```bash
export HUGGING_FACE_CACHE_DIR=/path/to/your/cache/dir
```

## :wrench: Inference Hugging Face Checkpoints

SDXL inference
```bash
PYTHONPATH=$(pwd) python inference_scripts/inference_spo_sdxl.py
```

SD v1.5 inference
```bash
PYTHONPATH=$(pwd) python inference_scripts/inference_spo_sd-v1-5.py
```

## :wrench: Training
The following scripts assume the use of **four** 80GB A100 GPUs for fine-tuning, as described in the [paper](https://arxiv.org/abs/2406.04314).

Before fine-tuning, please download the checkpoints of step-aware preference models. You can do this by following these steps:
```bash
sudo apt update
sudo apt install wget

mkdir model_ckpts
cd model_ckpts

wget https://huggingface.co/SPO-Diffusion-Models/Step-Aware_Preference_Models/resolve/main/sd-v1-5_step-aware_preference_model.bin

wget https://huggingface.co/SPO-Diffusion-Models/Step-Aware_Preference_Models/resolve/main/sdxl_step-aware_preference_model.bin

cd ..
```

To fine-tune SD v1.5, you can use the following command:
```bash
PYTHONPATH=$(pwd) accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_spo.py --config configs/spo_sd-v1-5_4k-prompts_num-sam-4_10ep_bs10.py
```
To fine-tune SDXL, you can use the following command:
```bash
PYTHONPATH=$(pwd) accelerate launch --config_file accelerate_cfg/1m4g_fp16.yaml train_scripts/train_spo_sdxl.py --config configs/spo_sdxl_4k-prompts_num-sam-2_3-is_10ep_bs2_gradacc2.py
```
To fine-tune using step-aware preference model checkpoints you’ve trained with the code in [step_aware_preference_model](https://github.com/RockeyCoss/SPO/tree/main/step_aware_preference_model), you can simply update the `config.preference_model_func_cfg.ckpt_path` setting in the config file to point to your desired checkpoint path. For example, you can modify [this line](https://github.com/RockeyCoss/SPO/blob/main/spo_training_and_inference/configs/spo_sdxl_4k-prompts_num-sam-2_3-is_10ep_bs2_gradacc2.py#L15) in the SDXL fine-tuning config.

## :unlock: Available Checkpoints

[SPO-SDXL_4k-prompts_10-epochs](https://huggingface.co/SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep)

[SPO-SDXL_4k-prompts_10-epochs_LoRA](https://huggingface.co/SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA)

[SPO-SD-v1-5_4k-prompts_10-epochs](https://huggingface.co/SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep)

[SPO-SD-v1-5_4k-prompts_10-epochs_LoRA](https://huggingface.co/SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep_LoRA)

## :mailbox_with_mail: Citation
If you find this code useful in your research, please consider citing:

```
@article{liang2024step,
  title={Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization},
  author={Liang, Zhanhao and Yuan, Yuhui and Gu, Shuyang and Chen, Bohan and Hang, Tiankai and Cheng, Mingxi and Li, Ji and Zheng, Liang},
  journal={arXiv preprint arXiv:2406.04314},
  year={2024}
}
```
