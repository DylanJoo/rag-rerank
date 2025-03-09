#!/bin/sh
#SBATCH --job-name=crux-eval
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate rag

# root
cd /home/dju/rag-rerank/src 

judge_model=meta-llama/Llama-3.1-70B-Instruct
mkdir -p logs/$judgement_model

# for result_file in results/testb-bm25*; do
#     file_name=${result_file##*/}
#     output_file=logs/$judge_model/${file_name/jsonl/log}
#     echo "Evaluating: " $file_name
#
#     python3 -m evaluation.llmjudge.retrieval_augmentation_generation \
#         --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.jsonl \
#         --corpus_dir /home/dju/datasets/crux/passages/ \
#         --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt \
#         --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
#         --model_name_or_path $judge_model \
#         --threshold 3 \
#         --gamma 0.5  \
#         --num_gpus 4 \
#         --result_jsonl $result_file > $output_file
# done

for result_file in results/testb-contriever*; do
    file_name=${result_file##*/}
    output_file=logs/$judge_model/${file_name/jsonl/log}
    echo "Evaluating: " $file_name

    python3 -m evaluation.llmjudge.retrieval_augmentation_generation \
        --topic_file /home/dju/datasets/crux/ranking_3/testb_topics.jsonl \
        --corpus_dir /home/dju/datasets/crux/passages/ \
        --qrels_file /home/dju/datasets/crux/ranking_3/testb_qrels_pr.txt \
        --judgement_file /home/dju/datasets/crux/ranking_3/testb_oracle-passages_judgements.jsonl \
        --model_name_or_path $judge_model \
        --threshold 3 \
        --gamma 0.5  \
        --num_gpus 4 \
        --result_jsonl $result_file > $output_file
done
