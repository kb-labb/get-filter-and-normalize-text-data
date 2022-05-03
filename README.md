# get-filter-and-normalize-text-data

This repository contains the methods to download, filter, and normalize the text data, mainly from KB's digitalized newspapers.

## Packages

To run the the various functions to download, filter, and normalize the data you will need to install and build some packages.

- `kenlm` https://github.com/kpu/kenlm
- `lsh` https://github.com/mattilyra/LSH
  - you might have to change the `setup.py` file: `USE_CYTHON = True` and `extensions = cythonize(extensions, force=True)`
- `unidecode`
- `tokenizers`
- `sentence_splitter`
- `fasttext`
  - download also the `lid.176.bin` [model](https://fasttext.cc/docs/en/language-identification.html)
- `kblab` https://github.com/Kungbib/kblab

## Usage

### Get the data

To download the data of one specific type/tag:

```bash
python get_data.py --tag <tag> --location <download_folder> --login_file <login_config.yaml>
```

Some of the available tags are:

- `magazine`
- `SOU`
- `protokoll`
- `proposition`
- `issue`

The data is downloaded into subfolders named after their _tag_ with all larger
document collections (e.g. one scanned newspaper issue) saved as one json-object
per line, sorted into one file for each publication year.
The extracted text boxes, that we use as documents, are stored in the
json-object accessible via the key `content`.

```
data
├── issue
│   ├── 1995.jsonl
│   ├── 1996.jsonl
│   ├── 1997.jsonl
│   ├── 1998.jsonl
├── SOU
│   ├── 1995.jsonl
│   ├── 1996.jsonl
│   ├── 1997.jsonl
│   ├── 1998.jsonl
...
```

### Filter & Normalize & Filter

The `clean_data.py` script is used to filter and normalize the downloaded json-files keeping the json-format intact.
The `filter_normalize_filter.sh` script first filters the data, to reduce the load for the normalizer, to the filter again after normalizing.

The various filter and normalizers that can be used are defined in `data_filters.py` and `data_normalizers.py`.

### Deduplication

The deduplication runs in three steps:

- `find_duplicates.py` finding which documents are duplicates of each other
- `kenlm_score.py` scoring everyone of these documents with a KenLM model
- `choose_best_duplicate.py` finally choosing the best duplicate with respect to the KenLM score

### Reformat

To get rid of the meta-information stored in the json-object, run again
`clean_data.py` with the `json2txt` flag, creating a text file with one document
per line.
