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

