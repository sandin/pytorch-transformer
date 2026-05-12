# pytorch-transformer
Attention is all you need implementation

YouTube video with full step-by-step implementation: https://www.youtube.com/watch?v=ISNdQcPhsts

## Vocabulary HTML export

The tokenizer files in `vocab/` can be exported to searchable HTML tables for easier inspection.

Run the exporter from the project root:

```bash
python scripts/export_vocab_html.py
```

By default, the script reads every `*.json` file in `vocab/` and writes the generated files to `vocab/html/`.
