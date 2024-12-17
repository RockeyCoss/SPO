PYTHONPATH=$(pwd) accelerate launch --dynamo_backend no --gpu_ids all \
--num_processes 4 --num_machines 1 --use_deepspeed \
trainer/scripts/train_spm.py +experiment=clip_h \
accelerator=sdxl_deepspeed \
task=sdxl_spm \
dataset=pick_a_pic_spm_sdxl \
optimizer=sdxl_dummy \
output_dir=work_dirs/sdxl_spm 
