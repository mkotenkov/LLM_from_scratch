RUN_NAME="transformer_uncased_2000_steps"

python3 /Users/maksimkoltugin/Dev/huawei_LLM_test_task/scripts/train_transformer.py \
--dataset_base_dir="/Users/maksimkoltugin/Dev/huawei_LLM_test_task/data/uncased-15k-10k" \
--ckpts_dir="/Users/maksimkoltugin/Dev/huawei_LLM_test_task/checkpoints/${RUN_NAME}/ckpts" \
--logs_dir="/Users/maksimkoltugin/Dev/huawei_LLM_test_task/checkpoints/${RUN_NAME}/logs" \
--tokenizer_path="/Users/maksimkoltugin/Dev/huawei_LLM_test_task/checkpoints/tokenizer/tokenizer_15k_10k_uncased.pkl"
