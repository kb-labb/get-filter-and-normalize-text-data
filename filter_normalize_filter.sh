#! /usr/bin/bash
set -ex

FILE=$1
NCPU=40
CHUNKSIZE=10


pre_filter="python clean_data.py \
                    --file $FILE \
                    --filter_by_language \
                    --filter_by_num_tokens \
                    --filter_tv_tables \
                    --chunksize $CHUNKSIZE \
                    --do_parallel \
                    --n_processes $NCPU"

normalize="python clean_data.py \
                    --file $FILE.filtered
                    --unicode_normalize \
                    --anonymize \
                    --common_errors \
                    --strip_incomplete_string \
                    --chunksize $CHUNKSIZE \
                    --do_parallel \
                    --n_processes $NCPU"

post_filter="python clean_data.py \
                    --file $FILE.filtered.normalized \
                    --filter_exact_duplicates \
                    --chunksize $CHUNKSIZE \
                    --n_processes $NCPU"

# Deduplication section

INPUT="$FILE.filtered.normalized.filtered"
DP_OUT="$INPUT.duplicate_candidates"
KENLM_OUT="$INPUT.kenlm_scores"
KENLM_MODEL="oscar+wiki.arpa.bin"
TOKENIZER="tokenizer.json"
DDUPED="$INPUT.deduped"
SEED="666"

    # --load-fingerprints
    # --save-fingerprints
fd_cmd="python find_duplicates.py \
    --seed $SEED \
    --inputs $INPUT \
    --output $DP_OUT \
    --jaccard union \
    --heuristic-iter 1 \
    --num-bands 10 \
    --num-seeds 100 \
    --num-workers $NCPU \
    --jaccard-parallel \
    --keep_doc_in_mem \
        "

kenlm_cmd="python kenlm_score.py \
    --input $INPUT \
    --output $KENLM_OUT \
    --num-workers $NCPU \
    --tokenizer $TOKENIZER \
    --kenlm-model $KENLM_MODEL \
        "

cbd_cmd="python choose_best_duplicate.py \
    --input $INPUT \
    --output $DDUPED \
    --scores $KENLM_OUT \
    --duplicate-candidates $DP_OUT \
        "

# deduplicate="$fd_cmd && $kenlm_cmd && $cbd_cmd"
###

# sent_split="python clean_data.py \
#                     --file $FILE \
#                     --sentence_split \
#                     --strip_incomplete_sentence \
#                     --n_processes"

# if you ran sent_split you need to change the input-file name by adding another .normalized
json2text="python clean_data.py \
                    --file $DDUPED \
                    --json2txt"


echo "Pre-Filter $FILE"
$pre_filter

echo "Normalize $FILE"
$normalize

echo "Post-Filter $FILE"
$post_filter

echo "Deduplicate $FILE"
$fd_cmd
$kenlm_cmd
$cbd_cmd

echo "Json2Text $FILE"
$json2text

exit 0
