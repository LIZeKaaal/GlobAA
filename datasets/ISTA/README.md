# Madelon training data

- Creator: **Isabelle Guyon**
- Source: [Madelon, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/171/madelon)
- DOI: [10.24432/C5602H](https://doi.org/10.24432/C5602H)
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Suggested dataset citation:

> Guyon, I. (2004). Madelon [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5602H.

## Included files

| Local filename | UCI filename | Shape |
| --- | --- | --- |
| `madelon_train.data.txt` | `MADELON/madelon_train.data` | 2,000 rows, 500 columns |
| `madelon_train.labels.txt` | `MADELON/madelon_train.labels` | 2,000 labels |

The bundled files retain the original data and add a `.txt` suffix to the
filenames. The dataset is distributed under CC BY 4.0. Preserve its attribution
and license link when redistributing the data, and identify any subsequent
changes.

## Use in the ENR/ISTA experiment

`3.2_ENR_ISTA/run_ENR_ISTA.py` loads the training matrix from this directory.
It centers and standardizes its columns in memory, and creates a synthetic
regression response using the experiment's random seed. These runtime operations
do not rewrite the dataset files. The supplied class labels are not used by the
ENR/ISTA experiment.

If the files are obtained again from UCI, place the training matrix here using
the exact local filename `madelon_train.data.txt`.
