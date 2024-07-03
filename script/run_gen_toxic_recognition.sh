#!/bin/bash
port=$(shuf -i 6000-9000 -n 1)
echo $port
# source /p/finetunellm/anaconda3/bin/activate /p/finetunellm/anaconda3
# conda activate fairbench

#### use_online_model=True
# model_name='chatgpt'
model_name='gpt-4'
#### need to set --online_model to activate the online model

# model_name='meta-llama/Llama-2-7b-chat-hf'
# model_name='meta-llama/Llama-2-13b-chat-hf'
# model_name='meta-llama/Llama-2-70b-chat-hf'
# model_name='meta-llama/Meta-Llama-3-8B-Instruct'
# model_name='meta-llama/Meta-Llama-3-70B-Instruct'
# model_name='mistralai/Mistral-7B-Instruct-v0.1'

task_type='toxicity_recognition'

export CUDA_VISIBLE_DEVICES=3

### Run the following command when you want to use the online model
python run_gen.py \
    --model_path ${model_name} \
    --test_type ${task_type} \
    --online_model \
### Run the following command when you want to use the offline model
# python run_gen.py \
#     --model_path ${model_name} \
#     --test_type ${task_type} \