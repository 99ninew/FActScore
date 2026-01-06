set -euo pipefail

export FACTSCORE_SAVE_EVERY=200

FILE_NAMES=(
     # "Bio-test_wikipage-quality-append-lr3e-06-rr0.9-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B" #plus_wikidata
     # "Bio-test_wikipage-quality-append-lr6e-06-rr0.9-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B"
     "Bio-test_wikipage-quality-append-lr8e-06-rr0.9-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B"
     "Bio-test_mix_wikidata_3+9_shuffle-quality-append-lr2e-05-rr0.9-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
     "Bio-test_mix_wikidata_6+6_shuffle-quality-append-lr2e-05-rr0.9-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
     "Bio-test_mix_wikidata_9+3_shuffle-quality-append-lr2e-05-rr0.9-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
     # "Bio-test_wikidata_x12-quality-append-lr2e-05-rr0.9-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
     "Bio-test_wikipage-quality-append-lr1e-05-rr0.9-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B"
     # "Bio-test_mix_wikidata_1+7_shuffle-quality-append-lr2e-05-rr0.3-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
     # "Bio-test_mix_wikidata_2+6_shuffle-quality-append-lr2e-05-rr0.3-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
     # "Bio-test_mix_wikidata_6+6_shuffle-quality-append-lr2e-05-rr0.3-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
     # "Bio-test_mix_wikidata_9+3_shuffle-quality-append-lr2e-05-rr0.3-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
     "Bio-test_wikipage-quality-append-lr1e-05-rr0.3-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B"
     "Bio-test_wikipage-quality-append-lr1.5e-05-rr0.3-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B"
     "Bio-test_wikipage-quality-append-lr6e-06-rr0.1-epochs8-bs8-wd0.01-warmup0.05-MetaLlama38B"
     # "Bio-test_mix_wikidata_1+7_shuffle-quality-append-lr2e-05-rr0.1-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
     # "Bio-test_mix_wikidata_2+6_shuffle-quality-append-lr2e-05-rr0.1-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
     "Bio-test_mix_wikidata_3+9_shuffle-quality-append-lr2e-05-rr0.1-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
     "Bio-test_mix_wikidata_9+3_shuffle-quality-append-lr2e-05-rr0.1-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
     "Bio-test_mix_wikidata_6+6_shuffle-quality-append-lr2e-05-rr0.1-epochs1-bs8-wd0.01-warmup0.05-MetaLlama38B"
)

for FILE_NAME in "${FILE_NAMES[@]}"; do
     echo "Processing ${FILE_NAME}..."
     LOG_PATH="scoring_logs/${FILE_NAME}.log"
     echo "=== START ${FILE_NAME} $(date) ===" | tee -a "${LOG_PATH}"

     # allow the command to fail without exiting the whole script so we can capture RC
     set +e
     python -m factscore.factscorer --input_path "out-less/${FILE_NAME}.jsonl" \
          --model_name retrieval+ChatGPT+npm --openai_key .cache/factscore/api_key.txt \
          --abstain_detection generic --verbose \
          --cache_dir ".cache/factscore/process/${FILE_NAME}" 
     RC=${PIPESTATUS[0]}
     set -e

     echo "=== END ${FILE_NAME} $(date) EXIT=${RC} ===" | tee -a "${LOG_PATH}"
     if [ ${RC} -ne 0 ]; then
        echo "Command failed with exit code ${RC} for ${FILE_NAME}; see ${LOG_PATH}" >&2
     fi
done
# python -m factscore.factscorer --input_path .cache/factscore/data/unlabeled/InstructGPT.jsonl \
#           --model_name retrieval+ChatGPT+npm --openai_key .cache/factscore/api_key.txt \
#           --abstain_detection generic --verbose \
#           --cache_dir ".cache/factscore/process/InstructGPT_our_pipe"

# bm25 cache files :
# stores topk related paragraphs for each atom fact in the corresponding-topic wikipedia page.
# npm cache files :
# stores the results (p_support) from the NPM model for each atom fact and its related paragraphs.