#! /usr/bin/bash
set -ex

FILE=$1
NCPU=20
CHUNKSIZE=100


split $1 $1. -d --line-bytes=10GB

for FILE in $1.*;
do


    pre_filter="python clean_data.py \
                        --file $FILE \
                        --filter_by_language \
                        --chunksize $CHUNKSIZE \
                        --n_processes $NCPU"
                        # --filter_by_num_tokens \
                        # no filtering by number of tokens to keep documents

    normalize="python clean_data.py \
                        --file $FILE.filtered
                        --unicode_normalize \
                        --anonymize \
                        --chunksize $CHUNKSIZE \
                        --n_processes $NCPU"

    # post filter finds exact duplicates, which is less efficient due to the processes
    # having to communicate with each other
    post_filter="python clean_data.py \
                        --file $FILE.filtered.normalized \
                        --filter_exact_duplicates \
                        --chunksize $CHUNKSIZE \
                        --n_processes $NCPU"
                        # --filter_by_unicode \
                        # ignore that for now and maybe add a normalize-function later


    fuse_paragraphs="python clean_data.py \
                            --file $FILE.filtered.normalized.filtered \
                            --fuse-paragraphs"
                            # --ignore-breaks \

    post_fuse_filter="python clean_data.py \
                        --file $FILE.filtered.normalized.filtered.fused \
                        --filter_by_num_tokens \
                        --remove_filter_breaks \
                        --chunksize $CHUNKSIZE \
                        --n_processes $NCPU"

    # Deduplication section
    # 

    INPUT="$FILE.filtered.normalized.filtered.fused.filtered"
    DP_OUT="$INPUT.duplicate_candidates"
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
            "


    cbd_cmd="python choose_best_duplicate.py \
        --input $INPUT \
        --output $DDUPED \
        --duplicate-candidates $DP_OUT \
            "

    deduplicate="$fd_cmd && $cbd_cmd"
    ###


    # if you ran sent_split you need to change the input-file name by adding another .normalized
    json2text="python clean_data.py \
                        --file $DDUPED \
                        --json2txt"

    # here we run the functions

    echo "Pre-Filter $FILE"
    $pre_filter
    
    echo "Normalize $FILE"
    $normalize
    
    echo "Post-Filter $FILE"
    $post_filter
    
    echo "fuse-paragraphs $FILE"
    $fuse_paragraphs

    echo "Post-fuse-Filter $FILE"
    $post_fuse_filter

    echo "Deduplicate $FILE"
    eval $deduplicate

    echo "Json2Text $FILE"
    $json2text
done

exit 0
