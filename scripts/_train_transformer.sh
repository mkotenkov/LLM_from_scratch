RUN_NAME="transformer_uncased_5000_steps"

python3 TEST_TASK_LLM/scripts/train_transformer.py \
--dataset_base_dir="TEST_TASK_LLM/data/uncased-15k-10k" \
--ckpts_dir="TEST_TASK_LLM/checkpoints/${RUN_NAME}/ckpts" \
--logs_dir="TEST_TASK_LLM/checkpoints/${RUN_NAME}/logs" \
--tokenizer_path="/data/d2/m.koltyugin/TEST_TASK_LLM/checkpoints/tokenizer/tokenizer_15k_10k_uncased.pkl"
