# As-Built Comparer

This program ingests a directory of PDFs that each contain
a single landscape-oriented as-built. It compares each as-built
to all other as-builts and exports a list of each
pair of PDF names along with a similarity score between them.
The higher the score, the more similar the as-builts.

## Installation

This program converts PDFs to images. For that process,
[pdf2image](https://github.com/Belval/pdf2image) is used.

`poppler` is required for this installation to be complete.
Please refer to pdf2image's documentation for more details.

After `poppler` is installed, run the following in a virtual
environment and with your shell's working directory set to the
root of this repo.

```shell
$ pip install -r requirements.txt
```

## Usage

To use this tool:

1. Ensure the virtual environment into which the dependencies have
been install is activated.
2. Ensure your shell's working directory is set to the root of this repo.
3. Run the following...

```shell
$  python compare.py /path/to/pdf/directory > results.csv
```

A message like the following will be printed to stderr.

```
Found 2 PDFs. Approximate load time: 8 seconds.
Loading images .. Finished.
Comparing 2 images. Approximate comparison time: 8 seconds.
Comparing images . Finished.
```

The results.csv file will contain the as-built names and similarity score.

