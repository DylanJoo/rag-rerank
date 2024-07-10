for split in dev test;do
    python3 tools/plaidx_search.py \
        --data_jsonl data/neuclir/requests-${split}-all.jsonl \
        --rewritten_json data/neuclir/gptqr-${split}-all.json \
        --output_tsv data/neuclir/hits-${split}-all.tsv \
        --top_k 30
done


