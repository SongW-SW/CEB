#!/bin/bash
port=$(shuf -i 6000-9000 -n 1)
echo $port

#### use_online_model=True
# model_name='chatgpt'
#model_name='gpt-4'
#### need to set --online_model to activate the online model

#model_name='meta-llama/Llama-2-7b-chat-hf'
#model_name='meta-llama/Llama-2-13b-chat-hf'
# model_name='meta-llama/Llama-2-70b-chat-hf'
#model_name='meta-llama/Meta-Llama-3-8B-Instruct'
# model_name='meta-llama/Meta-Llama-3-70B-Instruct'
model_name='mistralai/Mistral-7B-Instruct-v0.1'

task_type='stereotype_conversation'


python run_gen.py \
    --model_path ${model_name} \
    --test_type ${task_type} \

# python run_gen.py \
#     --model_path ${model_name} \
#     --test_type ${task_type} \