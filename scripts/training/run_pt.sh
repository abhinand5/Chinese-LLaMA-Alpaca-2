# 运行脚本前请仔细阅读wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/pt_scripts_zh)
# Read the wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/pt_scripts_zh) carefully before running the script
lr=2e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=/home/abhinand/projects/Chinese-LLaMA-Alpaca-2/models/llms/llama2-7b-hf
chinese_tokenizer_path=/home/abhinand/projects/Chinese-LLaMA-Alpaca-2/scripts/merge_tokenizer/merged_tokenizer_hf
dataset_dir=/home/abhinand/projects/Chinese-LLaMA-Alpaca-2/data/culturax_tamil_text
data_cache=/home/abhinand/projects/Chinese-LLaMA-Alpaca-2/data/cache_dir
per_device_train_batch_size=64
gradient_accumulation_steps=1
block_size=512
output_dir=/home/abhinand/projects/Chinese-LLaMA-Alpaca-2/models/llms/llama2-7b-tamil

deepspeed_config_file=ds_zero2_no_offload.json

torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.75 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --seed $RANDOM \
    --bf16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 2 \
    --save_steps 100 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype bfloat16 \
    --load_in_kbits 16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
# --resume_from_checkpoint /home/abhinand/projects/Chinese-LLaMA-Alpaca-2/models/llms/llama2-7b-tamil/checkpoint-400
