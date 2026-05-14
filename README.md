# pytorch-transformer
Attention is all you need implementation

YouTube video with full step-by-step implementation: https://www.youtube.com/watch?v=ISNdQcPhsts


## Datasets

To avoid network issues, you can download dataset file from [huggingface - opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books/resolve/main/en-it/train-00000-of-00001.parquet?download=true) to your local file: `dataset/train-00000-of-00001.parquet (5.73 MB)` .


## Build

```bash
uv sync
```


## Train


```bash
uv run python main.py

# or
uv run python main.py --config configs/opus_books.json
```

use tensorboard to minitor the training loss:
```bash
uv run tensorboard --logdir runs/tmodel
```
then open [http://localhost:6006/](http://localhost:6006/)


All weight files will be saved to `<config.model_folder>/<config.datasource>/<config.model_basename><epoch>.pt`


## Inference

```bash
uv run python translate.py "Who are you?"
uv run python translate.py "Who are you?" --config configs/opus_books.json

#  PREDICTED: Chi siete ?  % 
```


## Vocabulary HTML export

The tokenizer files in `vocab/` can be exported to searchable HTML tables for easier inspection.

Run the exporter from the project root:

```bash
uv run python scripts/export_vocab_html.py
```

By default, the script reads every `*.json` file in `vocab/` and writes the generated files to `vocab/html/`.
