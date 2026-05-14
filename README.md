# pytorch-transformer
Attention is all you need implementation

YouTube video with full step-by-step implementation: https://www.youtube.com/watch?v=ISNdQcPhsts


## Datasets

To avoid network issues, you can download dataset file from [huggingface - opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books/resolve/main/en-it/train-00000-of-00001.parquet?download=true) to your local file: `dataset/opus_books/train-00000-of-00001.parquet (5.73 MB)` .

Training datasets:
* [opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books) (en-it)
* [wmt19](https://huggingface.co/datasets/wmt/wmt19) (en-zh) (default)
* [Chinese_modern_classical](https://huggingface.co/datasets/xmj2002/Chinese_modern_classical)


## Build

```bash
uv sync
```


## Train


```bash
# use default config: wmt19.json
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
# or
uv run python translate.py "Who are you?" --config configs/opus_books.json

#  PREDICTED: Chi siete ?  % 
```

Or you can translate with BLEU score:

```bash
uv run python translate.py 2001  # datasets index 
```
Output:
```
Device name: mps
Loading Model: weights/wmt19/tmodel_87.pt
Loading local dataset from dataset/wmt19: 2 parquet file(s)
        ID: 2001
    SOURCE: Work you've done here is outstanding.
    TARGET: 你的工作完成的很出色。
 PREDICTED: 你的工作 完成的 很 出色 。  
      BLEU: 100.00
```


## GUI

Start the Gradio GUI:

```bash
uv run python ui.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860).

The GUI lets you select a config file from `configs/`, shows the source and target languages, and translates each message with `translate.py`.


## Vocabulary HTML export

The tokenizer files in `vocab/` can be exported to searchable HTML tables for easier inspection.

Run the exporter from the project root:

```bash
uv run python scripts/export_vocab_html.py
```

By default, the script reads every `*.json` file in `vocab/` and writes the generated files to `vocab/html/`.
