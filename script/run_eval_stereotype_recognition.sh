#!/bin/bash
data_path=./data/fairness/stereotype_recognition.json

port=$(shuf -i 6000-9000 -n 1)
echo $port

model_name='chatgpt'


python run_eval.py \
    --model_name_or_path ${model_name} \
    --stereotype_recognition_data_json_path ${data_path} \
