# Lab - Setup a project from scratch

## Run locally or in Colab

Install the dependencies and prepare the dataset:

```bash
python -m pip install -r requirements.txt
python utils/download_dataset.py
```

The download script also prepares the validation split for `ImageFolder`.

Train and evaluate:

```bash
python train.py
python eval.py
```

The dataset and generated checkpoints are kept locally and are not committed
to GitHub.
