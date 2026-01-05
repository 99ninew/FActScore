# TOKENIZER_PATH="~/CKM/model/Meta-Llama-3-8b"

# lower lr

# MODEL_PATH="~/CKM/ckpts/1223/test_wikipage-quality-append-lr3e-06-rr0.9-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-2376"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1223/test_wikipage-quality-append-lr6e-06-rr0.9-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-2376"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1223/test_wikipage-quality-append-lr8e-06-rr0.9-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-2376"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"



# mix=0.1 must
# MODEL_PATH="~/CKM/ckpts/1203/test_mix_wikidata_3+9_shuffle-quality-append-lr2e-05-rr0.9-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-3562"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1203/test_mix_wikidata_6+6_shuffle-quality-append-lr2e-05-rr0.9-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-3562"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1203/test_mix_wikidata_9+3_shuffle-quality-append-lr2e-05-rr0.9-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-3562"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1203/test_wikidata_x12-quality-append-lr2e-05-rr0.9-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-3562"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1119/test_wikipage-quality-append-lr1e-05-rr0.9-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-2376"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"


# val_loss<0.06
# mix=0.9
# Test it at last!!, after scoring all the other ckpts. 
# MODEL_PATH="~/CKM/ckpts/1115/test_mix_wikidata_3+9_shuffle-quality-append-lr2e-05-rr0.1-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-396"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1115/test_wikipage-quality-append-lr6e-06-rr0.1-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-264"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1113/test_mix_wikidata_1+7_shuffle-quality-append-lr2e-05-rr0.1-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-264"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# # in 1113/ so did not generate
# MODEL_PATH="~/CKM/ckpts/1115/test_mix_wikidata_2+6_shuffle-quality-append-lr2e-05-rr0.1-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-264"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1115/test_mix_wikidata_9+3_shuffle-quality-append-lr2e-05-rr0.1-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-396"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1115/test_mix_wikidata_6+6_shuffle-quality-append-lr2e-05-rr0.1-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-396"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# mix=0.7
# MODEL_PATH="~/CKM/ckpts/1119/test_mix_wikidata_1+7_shuffle-quality-append-lr2e-05-rr0.3-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-340"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1119/test_mix_wikidata_2+6_shuffle-quality-append-lr2e-05-rr0.3-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-340"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1119/test_mix_wikidata_6+6_shuffle-quality-append-lr2e-05-rr0.3-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-509"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1119/test_mix_wikidata_9+3_shuffle-quality-append-lr2e-05-rr0.3-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-509"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1119/test_wikidata_x12-quality-append-lr2e-05-rr0.3-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-509"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# MODEL_PATH="~/CKM/ckpts/1119/test_wikipage-quality-append-lr1e-05-rr0.3-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-344"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# # val_loss > 0.06
# MODEL_PATH="~/CKM/ckpts/1119/test_wikipage-quality-append-lr1.5e-05-rr0.3-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-344"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"

# mix=0.3
# MODEL_PATH="~/CKM/ckpts/1207/test_wikipage-quality-append-lr1e-05-rr0.7-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B/checkpoint-792"
# python generate_outputs_local.py --model_type vllm --model_path="$MODEL_PATH"