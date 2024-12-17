# Step-Aware Preference Model Training Code

This folder contains the code for training the **step-aware preference model**. The codebase is based on [PickScore](https://github.com/yuvalkirstain/PickScore).

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
cd ./SPO/step_aware_preference_model
```
4. Install Dependencies
```bash
pip uninstall peft -y
pip install -r requirements.txt
```
5. Login to Weights & Biases (wandb)
```bash
wandb login {Your wandb key}
```
6. (Optional) To customize the location for saving models downloaded from Hugging Face, you can use the following command:
```bash
export HUGGING_FACE_CACHE_DIR=/path/to/your/cache/dir
```
## Download the Pick-a-Pic Dataset
```bash
from datasets import load_dataset
dataset = load_dataset("yuvalkirstain/pickapic_v1", num_proc=64)
``` 
For more details, please visit the [PickScore Github repository](https://github.com/yuvalkirstain/PickScore?tab=readme-ov-file#download-the-pick-a-pic-dataset).

## Training
The following scripts assume the use of **four** 80GB A100 GPUs for training, as described in the [paper](https://arxiv.org/abs/2406.04314).


To train the step-aware preference model for SD v1.5, please use the following command:
```bash
bash run_commands/train_spm_sd15.sh
```
To train the step-aware preference model for SDXL, please use the following command:
```bash
bash run_commands/train_spm_sdxl.sh
```
The final checkpoints, i.e., `work_dirs/sdv15_spm/final_ckpt.bin` and `work_dirs/sdxl_spm/final_ckpt.bin`, can be used for SPO training. Please refer to [this](https://github.com/RockeyCoss/SPO/blob/main/spo_training_and_inference/README.md#wrench-training) for more details.

## Citation
If you find this code useful in your research, please consider citing:

```
@article{liang2024step,
  title={Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization},
  author={Liang, Zhanhao and Yuan, Yuhui and Gu, Shuyang and Chen, Bohan and Hang, Tiankai and Cheng, Mingxi and Li, Ji and Zheng, Liang},
  journal={arXiv preprint arXiv:2406.04314},
  year={2024}
}
```
