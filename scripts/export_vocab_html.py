#!/usr/bin/env python3
"""Export tokenizer vocabulary JSON files to readable HTML tables."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any


def load_vocab(data: dict[str, Any]) -> dict[str, int]:
    """Return the vocabulary map from common tokenizer JSON layouts."""
    model_vocab = data.get("model", {}).get("vocab")
    if isinstance(model_vocab, dict):
        return model_vocab

    vocab = data.get("vocab")
    if isinstance(vocab, dict):
        return vocab

    raise ValueError("Could not find a vocabulary at 'model.vocab' or 'vocab'.")


def load_special_tokens(data: dict[str, Any]) -> set[str]:
    """Return token strings marked as special in the tokenizer JSON."""
    special_tokens: set[str] = set()
    for token in data.get("added_tokens", []):
        if isinstance(token, dict) and token.get("special") is True:
            content = token.get("content")
            if isinstance(content, str):
                special_tokens.add(content)
    return special_tokens


def render_html(source_path: Path, data: dict[str, Any]) -> str:
    vocab = load_vocab(data)
    special_tokens = load_special_tokens(data)
    rows = sorted(vocab.items(), key=lambda item: (item[1], item[0]))

    table_rows = "\n".join(
        "        <tr>"
        f"<td>{token_id}</td>"
        f"<td><code>{escape(token)}</code></td>"
        f"<td><code>{escape(repr(token))}</code></td>"
        f"<td>{'yes' if token in special_tokens else ''}</td>"
        "</tr>"
        for token, token_id in rows
    )

    model = data.get("model", {})
    model_type = model.get("type", "unknown") if isinstance(model, dict) else "unknown"
    unk_token = model.get("unk_token", "") if isinstance(model, dict) else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(source_path.name)} vocabulary</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #172033;
      --muted: #5d667a;
      --line: #d9e0ea;
      --accent: #1f7a8c;
      --head: #e9eef5;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #111827;
        --panel: #172033;
        --text: #f3f6fb;
        --muted: #aeb8c8;
        --line: #334155;
        --accent: #56b6c2;
        --head: #223047;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 32px auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 16px 0;
      color: var(--muted);
    }}
    .summary span {{
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
    }}
    label {{
      display: block;
      margin: 18px 0 10px;
      font-weight: 600;
    }}
    input {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 8px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: var(--head);
      color: var(--text);
      font-size: 13px;
      text-transform: uppercase;
    }}
    td:first-child {{
      width: 96px;
      color: var(--accent);
      font-variant-numeric: tabular-nums;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    tr[hidden] {{ display: none; }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(source_path.name)} vocabulary</h1>
    <div class="summary">
      <span>{len(rows):,} tokens</span>
      <span>model: {escape(str(model_type))}</span>
      <span>unk token: {escape(str(unk_token))}</span>
      <span>{len(special_tokens):,} special tokens</span>
    </div>
    <label for="filter">Filter tokens</label>
    <input id="filter" type="search" placeholder="Type to filter by token, repr, id, or special flag">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Token</th>
            <th>Python repr</th>
            <th>Special</th>
          </tr>
        </thead>
        <tbody>
{table_rows}
        </tbody>
      </table>
    </div>
  </main>
  <script>
    const filter = document.getElementById("filter");
    const rows = Array.from(document.querySelectorAll("tbody tr"));
    filter.addEventListener("input", () => {{
      const query = filter.value.trim().toLowerCase();
      for (const row of rows) {{
        row.hidden = query !== "" && !row.textContent.toLowerCase().includes(query);
      }}
    }});
  </script>
</body>
</html>
"""


def export_file(source_path: Path, output_dir: Path) -> Path:
    data = json.loads(source_path.read_text(encoding="utf-8"))
    output_path = output_dir / f"{source_path.stem}.html"
    output_path.write_text(render_html(source_path, data), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export vocab JSON files to human-readable HTML tables."
    )
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=Path("vocab"),
        help="Directory containing tokenizer JSON files. Defaults to ./vocab.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vocab/html"),
        help="Directory for generated HTML files. Defaults to ./vocab/html.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab_dir = args.vocab_dir
    output_dir = args.output_dir

    if not vocab_dir.is_dir():
        raise SystemExit(f"Vocabulary directory not found: {vocab_dir}")

    json_files = sorted(vocab_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in: {vocab_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for source_path in json_files:
        output_path = export_file(source_path, output_dir)
        print(f"Exported {source_path} -> {output_path}")


if __name__ == "__main__":
    main()
