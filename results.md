## Summary:
|Round | description                       | avg score | success rate | training size | eval size |
|------|-----------------------------------|-----------|--------------|---------------|-----------|
|1     | initial fine tune                 |   2.33    |   33.33%     |      40       |    9      |
|2     | with prompt tunning               |   1.44    |    100%      |      40       |    9      |
|3     | with prompt tunning and more data |   1.18    |    100%      |      277      |    70     |


### Round 1
```
> /mnt/d/health-training/vision_bcs$ python3 finetune_qwen3_vl_4b_lora.py
/mnt/d/health-training/vision_bcs/datasets/essay/dataset_updated.csv
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights:   0%|▏                                                                                                      | 1/713 [00:00<08:14,  1.44it/s]~/.local/lib/python3.10/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 713/713 [00:08<00:00, 81.39it/s]The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 151645, 'bos_token_id': None, 'pad_token_id': 151643}.
{'loss': '14.53', 'grad_norm': '3.516', 'learning_rate': '1e-05', 'entropy': '4.01', 'num_tokens': '3.098e+04', 'mean_token_accuracy': '0.08726', 'epoch': '1'}
{'train_runtime': '80.28', 'train_samples_per_second': '0.498', 'train_steps_per_second': '0.125', 'train_loss': '14.53', 'epoch': '1'}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:20<00:00,  8.03s/it]The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
~/.local/lib/python3.10/site-packages/bitsandbytes/backends/cuda/ops.py:468: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
saved_adapter= /mnt/d/health-training/vision_bcs/outputs/qwen3_vl_4b_lora_bcs_with_prompt_tunning
metrics= {"eval_count": 9, "parsed": 3, "coverage": 0.3333333333333333, "mae": 2.3333333333333335, "train_size": 40, "model_id": "Qwen/Qwen3-VL-4B-Instruct"}> /mnt/d/health-training/vision_bcs$ python3
```


### Round 2
```
> /mnt/d/health-training/vision_bcs$ python3 finetune_qwen3_vl_4b_lora.py
/mnt/d/health-training/vision_bcs/datasets/essay/dataset_updated.csv
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
model.safetensors.index.json: 64.7kB [00:00, 78.1MB/s]
Fetching 2 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [02:45<00:00, 82.74s/it]Download complete: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 8.88G/8.88G [02:45<00:00, 53.6MB/s]Loading weights:   0%|▏                                                                                                      | 1/713 [00:00<01:37,  7.30it/s]~/.local/lib/python3.10/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 713/713 [00:01<00:00, 374.98it/s]generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 269/269 [00:00<00:00, 1.12MB/s]The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 151645, 'bos_token_id': None, 'pad_token_id': 151643}.
{'loss': '10.41', 'grad_norm': '2.516', 'learning_rate': '1e-05', 'entropy': '3.13', 'num_tokens': '4.234e+04', 'mean_token_accuracy': '0.2484', 'epoch': '1'}
{'train_runtime': '130.8', 'train_samples_per_second': '0.306', 'train_steps_per_second': '0.076', 'train_loss': '10.41', 'epoch': '1'}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [02:10<00:00, 13.08s/it]The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
~/.local/lib/python3.10/site-packages/bitsandbytes/backends/cuda/ops.py:468: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
saved_adapter= /mnt/d/health-training/vision_bcs/outputs/qwen3_vl_4b_lora_bcs_with_prompt_tunning
metrics= {"eval_count": 9, "parsed": 9, "coverage": 1.0, "mae": 1.4444444444444444, "train_size": 40, "model_id": "Qwen/Qwen3-VL-4B-Instruct"}
```


### Round 3
```
> /mnt/d/health-training/vision_bcs$ python3 finetune_qwen3_vl_4b_lora.py
/mnt/d/health-training/vision_bcs/datasets/test/dataset.csv
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights:   0%|▏                                                                                                      | 1/713 [00:00<08:04,  1.47it/s]~/.local/lib/python3.10/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 713/713 [00:08<00:00, 84.46it/s]
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 151645, 'bos_token_id': None, 'pad_token_id': 151643}.
{'loss': '10.11', 'grad_norm': '0.8438', 'learning_rate': '8.714e-05', 'entropy': '4.553', 'num_tokens': '4.611e+04', 'mean_token_accuracy': '0.1968', 'epoch': '0.1444'}
{'loss': '6.162', 'grad_norm': '0.3457', 'learning_rate': '7.286e-05', 'entropy': '6.284', 'num_tokens': '9.429e+04', 'mean_token_accuracy': '0.2231', 'epoch': '0.2888'}
{'loss': '5.493', 'grad_norm': '0.1611', 'learning_rate': '5.857e-05', 'entropy': '5.698', 'num_tokens': '1.397e+05', 'mean_token_accuracy': '0.2783', 'epoch': '0.4332'}
{'loss': '5.261', 'grad_norm': '0.1016', 'learning_rate': '4.429e-05', 'entropy': '5.414', 'num_tokens': '1.86e+05', 'mean_token_accuracy': '0.3187', 'epoch': '0.5776'}
{'loss': '5.329', 'grad_norm': '0.06689', 'learning_rate': '3e-05', 'entropy': '5.455', 'num_tokens': '2.336e+05', 'mean_token_accuracy': '0.3139', 'epoch': '0.722'}
{'loss': '5.226', 'grad_norm': '0.05249', 'learning_rate': '1.571e-05', 'entropy': '5.341', 'num_tokens': '2.804e+05', 'mean_token_accuracy': '0.3289', 'epoch': '0.8664'}
{'loss': '5.14', 'grad_norm': '0.04932', 'learning_rate': '1.429e-06', 'entropy': '5.236', 'num_tokens': '3.228e+05', 'mean_token_accuracy': '0.3418', 'epoch': '1'}
{'train_runtime': '752.5', 'train_samples_per_second': '0.368', 'train_steps_per_second': '0.093', 'train_loss': '6.103', 'epoch': '1'}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 70/70 [12:32<00:00, 10.75s/it]
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
~/.local/lib/python3.10/site-packages/bitsandbytes/backends/cuda/ops.py:468: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
saved_adapter= /mnt/d/health-training/vision_bcs/outputs/qwen3_vl_4b_lora_bcs_with_prompt_tunning
metrics= {"eval_count": 70, "parsed": 70, "coverage": 1.0, "mae": 1.1857142857142857, "train_size": 277, "model_id": "Qwen/Qwen3-VL-4B-Instruct"}
```
