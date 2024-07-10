for split in dev test;do
    python3 tools/prepare_gpt_rewrite.py \
        --input_jsonl data/neuclir/requests-${split}-all.jsonl \
        --output_csv data/neuclir/qr-prompts-${split}-all.csv
done


