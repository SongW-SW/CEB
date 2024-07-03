sbatch --output=/p/finetunellm/LLM_evaluation/FairBench/slurm/%A_%a-%x.out \
    --error=/p/finetunellm/LLM_evaluation/FairBench/slurm/%A_%a-%x.err \
    -p gpu \
    -N 1 \
    --ntasks-per-node 1 \
    --mem=100G \
    --cpus-per-task 32 \
    --gres=gpu:a100:1 \
    --time 1:0:0 \
    --array 0-0 \
    --job-name eval_llama2_7b <<EOF
#!/bin/bash
PORT=\$(expr \$RANDOM + 1000) srun --wait 0 bash run_gen_toxic_recognition.sh
EOF