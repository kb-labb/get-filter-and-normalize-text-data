#! /usr/bin/bash
set -ex

FILE=$1
NCPU=40

pre_filter="python clean_data.py \
                    --file $FILE \
                    --filter_by_num_tokens \
                    --filter_by_language \
                    --n_processes $NCPU"

# normalize takes time due to --common_errors
# the regex has to take contexts into account and is therefore less efficient
normalize="python clean_data.py \
                    --file $FILE.filtered
                    --unicode_normalize \
                    --moses_normalize \
                    --anonymize \
                    --n_processes $NCPU"

# post filter finds exact duplicates, which is less efficient due to the processes
# having to communicate with each other
post_filter="python clean_data.py \
                    --file $FILE.filtered.normalized \
                    --filter_by_num_tokens \
                    --filter_by_language \
                    --filter_by_unicode \
                    --filter_exact_duplicates \
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
        "

cbd_cmd="python choose_best_duplicate.py \
    --input $INPUT \
    --output $DDUPED \
    --duplicate-candidates $DP_OUT \
        "

deduplicate="$fd_cmd && $cbd_cmd"
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
eval $deduplicate
# $cbd_cmd

echo "Json2Text $FILE"
# $json2text

exit 0
