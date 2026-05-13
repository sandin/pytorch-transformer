# pytorch-transformer
Attention is all you need implementation

YouTube video with full step-by-step implementation: https://www.youtube.com/watch?v=ISNdQcPhsts


## Build

```bash
uv sync
```


## Train


```bash
uv run python main.py
```

All weight files will be saved to `<dataset>_<config.model_folder>/<config.model_basename><epoch>.pt`


## Inference

```bash
uv run python translate.py "Who are you?"

#  PREDICTED: Chi siete ?  % 
```


## Vocabulary HTML export

The tokenizer files in `vocab/` can be exported to searchable HTML tables for easier inspection.

Run the exporter from the project root:

```bash
python scripts/export_vocab_html.py
```

By default, the script reads every `*.json` file in `vocab/` and writes the generated files to `vocab/html/`.
