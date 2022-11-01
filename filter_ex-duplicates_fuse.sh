#! /usr/bin/bash
set -ex

FILE=$1
NCPU=48


pre_filter="python clean_data.py \
                    --file $FILE \
                    --filter_by_language \
                    --n_processes $NCPU"
                    # --filter_by_num_tokens \
                    # no filtering by number of tokens to keep documents

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
                    --filter_by_language \
                    --filter_exact_duplicates \
                    --n_processes $NCPU"
                    # --filter_by_unicode \
                    # ignore that for now and maybe add a normalize-function later


# Deduplication section
# 

INPUT="$FILE.filtered.normalized.filtered"
FUSED_INPUT="$INPUT.fused"
DP_OUT="$FUSED_INPUT.duplicate_candidates"
KENLM_OUT="$FUSED_INPUT.kenlm_scores"
KENLM_MODEL="oscar+wiki.arpa.bin"
TOKENIZER="tokenizer.json"
DDUPED="$FUSED_INPUT.deduped"
SEED="666"

    # --load-fingerprints
    # --save-fingerprints
fd_cmd="python find_duplicates.py \
    --seed $SEED \
    --inputs $FUSED_INPUT \
    --output $DP_OUT \
    --jaccard union \
    --heuristic-iter 1 \
    --num-bands 10 \
    --num-seeds 100 \
    --num-workers $NCPU \
    --jaccard-parallel \
        "

cbd_cmd="python choose_best_duplicate.py \
    --input $FUSED_INPUT \
    --output $DDUPED \
    --duplicate-candidates $DP_OUT \
        "

# deduplicate="$fd_cmd && $kenlm_cmd && $cbd_cmd"

# we skip the kenlm document-scoring and simply take the longest document in 
# case of a LSH-base duplicate document
deduplicate="$fd_cmd && $cbd_cmd"
###


# if you ran sent_split you need to change the input-file name by adding another .normalized
json2text="python clean_data.py \
                    --file $DDUPED \
                    --json2txt"

fuse_paragraphs="python clean_data.py \
                        --file $INPUT \
                        --fuse-paragraphs"


# here we run the functions

echo "Pre-Filter $FILE"
$pre_filter

echo "Normalize $FILE"
$normalize

echo "Post-Filter $FILE"
$post_filter

echo "fuse-paragraphs $FILE"
$fuse_paragraphs

echo "Deduplicate $FILE"
eval $deduplicate

echo "Json2Text $FILE"
$json2text

exit 0
