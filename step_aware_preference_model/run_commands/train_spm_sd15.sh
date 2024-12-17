PYTHONPATH=$(pwd) accelerate launch --dynamo_backend no --gpu_ids all \
--num_processes 4 --num_machines 1 --use_deepspeed \
trainer/scripts/train_spm.py +experiment=clip_h
