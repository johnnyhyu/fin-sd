## You can check the batch status on openai dashboard.
## https://platform.openai.com/batches
python utils/openai_batch.py \
        --dataset "your_dataset" \
        --subset "your_subset" \
        --prompt "your_prompt_type" \
        --model "your_model_id" \
        --api_key "your_api_key" \
        --base_url "your_base_url"