import argparse
from typing import Iterable, Dict
from datasets import load_dataset, concatenate_datasets, Dataset
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
    processors,
    Regex,
)

PAD = "[PAD]"
BOS = "[CLS]"
EOS = "[SEP]"
MASK = "[MASK]"
UNK = "[UNK]"
replacement = "▁"
TOKENIZER_TYPES = ["unigram", "wordpiece", "bpe"]


def batch_iterator(dataset: Dataset, dataset_size: int,
                   batch_size: int) -> Iterable[Dict[str, str]]:
    for i in range(0, dataset_size, batch_size):
        yield dataset[i:i + batch_size]["text"]


# https://github.com/huggingface/tokenizers/issues/640#issuecomment-792305076
def tokenizer_trainer(text,
                      tok_type: str,
                      vocab_size: int,
                      tokenizer_file: str = "tokenizer.json",
                      min_frequency: int = 0,
                      add_prefix_space: bool = True,
                      batch_size: int = 50) -> None:
    # Supply either path to txt file or list of strings as text arg

    assert tok_type in TOKENIZER_TYPES

    # tokenizer = Tokenizer(models.WordPiece(unk_token=UNK))
    if tok_type == "unigram":
        tokenizer = Tokenizer(models.Unigram())

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Metaspace(replacement=replacement,
                                     add_prefix_space=add_prefix_space),
            pre_tokenizers.WhitespaceSplit(),  # does not split on punctuation
            pre_tokenizers.Split(Regex("\d"), behavior="merged_with_previous"),
            pre_tokenizers.Punctuation(),
        ])

        tokenizer.decoder = decoders.Metaspace()

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=[PAD, UNK, MASK, BOS, EOS],
            min_frequency=min_frequency,
            unk_token=UNK,
            shrinking_factor=0.75,  # 0.75
            max_piece_length=16,  # 16
            n_sub_iterations=2,  # 2
        )

    elif tok_type == "wordpiece":
        tokenizer = Tokenizer(models.WordPiece())

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),  # does not split on punctuation
            pre_tokenizers.Split(Regex("\d"), behavior="merged_with_previous"),
            pre_tokenizers.Punctuation(),
        ])

        tokenizer.decoder = decoders.WordPiece()

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=[PAD, UNK, MASK, BOS, EOS],
            min_frequency=min_frequency,
            unk_token=UNK,
        )

    if tok_type == "bpe":
        tokenizer = Tokenizer(models.BPE())

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space),
            pre_tokenizers.WhitespaceSplit(),  # does not split on punctuation
            pre_tokenizers.Split(Regex("\d"), behavior="merged_with_previous"),
            pre_tokenizers.Punctuation(),
        ])
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[PAD, UNK, MASK, BOS, EOS],
            min_frequency=min_frequency,
        )

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Nmt(),
        normalizers.NFKC(),
        # normalizers.NFD(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ])

    if isinstance(text, str):
        # if user specified path to txt file as string
        tokenizer.train(text, trainer=trainer)
    else:
        # text is a datasets Dataset
        tokenizer.train_from_iterator(batch_iterator(text, len(text),
                                                     batch_size),
                                      trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A {EOS}",
        pair=f"{BOS} $A {EOS} $B:1 {EOS}:1",
        special_tokens=[
            (f"{BOS}", tokenizer.token_to_id(BOS)),
            (f"{EOS}", tokenizer.token_to_id(EOS)),
        ],
    )
    tokenizer.save(tokenizer_file, pretty=True)
    # tokenizer.model.save("output_dir")
    return


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infiles", nargs="+", type=str)
    parser.add_argument("--tokenizer_name", type=str, default="tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--min_frequency", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--add_prefix_space", action="store_true")
    parser.add_argument("--tok_type",
                        choices=["bpe", "wordpiece", "unigram"],
                        default="wordpiece")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print("Loading datasets")
    dataset = load_dataset(
        "text",
        data_files={str(i): name
                    for i, name in enumerate(args.infiles)},
        cache_dir="cache_dataset",
    )
    print("Concatenating datasets")
    dataset = concatenate_datasets(
        [dataset[str(i)] for i, _ in enumerate(args.infiles)])
    print("Start training the tokenizer")
    tokenizer_trainer(text=dataset,
                      tok_type=args.tok_type,
                      vocab_size=args.vocab_size,
                      tokenizer_file=args.tokenizer_name,
                      min_frequency=args.min_frequency,
                      add_prefix_space=args.add_prefix_space,
                      batch_size=args.batch_size)
